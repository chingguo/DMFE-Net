import torch
import torch.nn as nn
import torch.nn.functional as F


class SFE(nn.Module):
    def __init__(self, dim):
        super(SFE, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_25 = nn.Conv2d(dim, dim, kernel_size=9, padding=12, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.cat([self.conv3_7(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        return x


class TFE(nn.Module):
    def __init__(self, dim):
        super(TFE, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.res = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 3, 1, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        ori = x
        x = self.res(x)
        x = ori + x
        x = self.conv(x)

        return x


class FFE(nn.Module):
    def __init__(self, dim):
        super(FFE, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)

    def forward(self, x):
        fft_size = x.size()[2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        x = x1 * x2_fft

        x = torch.fft.ifft2(x, norm='backward')
        x = torch.abs(x)

        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(feats_sum)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class MultiDomainExtraction(nn.Module):
    def __init__(self, dim):
        super(MultiDomainExtraction, self).__init__()
        self.norm = nn.BatchNorm2d(dim)

        self.SFE = SFE(dim)
        self.TFE = TFE(dim)
        self.FFE = FFE(dim)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.channel_adj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1)
        )

    def forward(self, x):
        ori = x
        x = self.norm(x)

        # x1 = self.TFE(x)
        x = self.SFE(x)
        # x3 = self.FFE(ori)

        # x = x1 + x2
        x = self.pa(x) * x
        x = x + ori
        # x = torch.cat([x, x3],dim=1)

        x = self.ca(x) * x
        x = self.channel_adj(x)
        x = x + ori
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [MultiDomainExtraction(dim=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Model(nn.Module):
    def __init__(self, dim_list=[24, 48, 96, 48, 24],
                 depth_list=[1, 1, 2, 1, 1]):
        super(Model, self).__init__()
        self.patch_size = 4
        self.inputconv = nn.Conv2d(3, 24, 3, 1, 1)
        self.outputconv = nn.Conv2d(24, 4, 3, 1, 1)

        self.block1 = BasicLayer(dim=dim_list[0], depth=depth_list[0])
        self.skip1 = nn.Conv2d(dim_list[0], dim_list[0], 1)

        self.downsample1 = nn.Conv2d(dim_list[0], dim_list[1], 3, 2, 1)

        self.block2 = BasicLayer(dim=dim_list[1], depth=depth_list[1])
        self.skip2 = nn.Conv2d(dim_list[1], dim_list[1], 1)

        self.downsample2 = nn.Conv2d(dim_list[1], dim_list[2], 3, 2, 1)

        self.block3 = BasicLayer(dim=dim_list[2], depth=depth_list[2])
        self.upsample1 = nn.Sequential(
            nn.Conv2d(dim_list[2], dim_list[3] * 4, 1, 1, padding_mode='reflect'),
            nn.PixelShuffle(2)
        )
        self.fusion1 = SKFusion(dim_list[3])

        self.block4 = BasicLayer(dim=dim_list[3], depth=depth_list[3])
        self.upsample2 = nn.Sequential(
            nn.Conv2d(dim_list[3], dim_list[4] * 4, 1, 1, padding_mode='reflect'),
            nn.PixelShuffle(2)
        )
        self.fusion2 = SKFusion(dim_list[4])

        self.block5 = BasicLayer(dim=dim_list[4], depth=depth_list[4])

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.inputconv(x)
        x = self.block1(x)
        skip1 = x

        x = self.downsample1(x)
        x = self.block2(x)
        skip2 = x

        x = self.downsample2(x)
        x = self.block3(x)
        x = self.upsample1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.block4(x)
        x = self.upsample2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.block5(x)
        x = self.outputconv(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat = self.forward_features(x)

        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x
