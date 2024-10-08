import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from utils import calculate_FFT
from utils.metrics import ssim, psnr
from model.model import Model
from dataset.loader import get_dir, Loader


def train(train_dataloader, model, criterion, optimizer, jundge_autocast, scaler):
    model.train()
    losses = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for batch in tqdm(train_dataloader, 'Training'):
        hazy_img = batch['source'].cuda()
        clear_img = batch['target'].cuda()
        with autocast(jundge_autocast):
            output = model(hazy_img)
            loss_s = criterion(output, clear_img)
            clear_fft = calculate_FFT(clear_img)
            output_fft = calculate_FFT(output)
            loss_f = criterion(output_fft, clear_fft)
            loss = loss_s + 0.01 * loss_f
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses += loss.item() * hazy_img.size(0)

        # [-1, 1] to [0, 1]
        output = output * 0.5 + 0.5
        clear_img = clear_img * 0.5 + 0.5

        running_psnr = 10 * torch.log10(1 / F.mse_loss(output, clear_img)).item()
        total_psnr += running_psnr

        _, _, H, W = output.size()
        down_ratio = max(1, round(min(H, W) / 256))

        count_ssim = 0.0
        count = 0
        for i in range(output.shape[0]):
            running_ssim = ssim(F.adaptive_avg_pool2d(output[i:i + 1], (int(H / down_ratio), int(W / down_ratio))),
                                F.adaptive_avg_pool2d(clear_img[i:i + 1], (int(H / down_ratio), int(W / down_ratio))),
                                data_range=1, size_average=False).item()
            count += 1
            count_ssim += running_ssim
        avg_ssim = count_ssim / count
        total_ssim += avg_ssim

        '''
        running_psnr = psnr(output, clear_img)
        total_psnr += running_psnr

        running_ssim = ssim(output, clear_img)  
        total_ssim += running_ssim.item()
        '''
        optimizer.zero_grad()

    return losses, total_psnr, total_ssim


def valid(valid_dataloader, model, criterion):
    model.eval()
    losses = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    for batch in tqdm(valid_dataloader, 'Validing'):
        hazy_img = batch['source'].cuda()
        clear_img = batch['target'].cuda()
        with torch.no_grad():
            output = model(hazy_img)
            loss_s = criterion(output, clear_img)
            clear_fft = calculate_FFT(clear_img)
            output_fft = calculate_FFT(output)
            loss_f = criterion(output_fft, clear_fft)
            loss = loss_s + 0.01 * loss_f

        losses += loss.item() * hazy_img.size(0)

        # [-1, 1] to [0, 1]
        output = output * 0.5 + 0.5
        clear_img = clear_img * 0.5 + 0.5

        running_psnr = 10 * torch.log10(1 / F.mse_loss(output, clear_img)).item()
        total_psnr += running_psnr

        _, _, H, W = output.size()
        down_ratio = max(1, round(min(H, W) / 256))

        count_ssim = 0.0
        count = 0
        for i in range(output.shape[0]):
            running_ssim = ssim(F.adaptive_avg_pool2d(output[i:i + 1], (int(H / down_ratio), int(W / down_ratio))),
                                F.adaptive_avg_pool2d(clear_img[i:i + 1], (int(H / down_ratio), int(W / down_ratio))),
                                data_range=1, size_average=False).item()
            count += 1
            count_ssim += running_ssim
        avg_ssim = count_ssim / count
        total_ssim += avg_ssim

    return losses, total_psnr, total_ssim


def test(test_dataloader, model, criterion):
    model.eval()
    losses = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    for batch in tqdm(test_dataloader, 'Testing'):
        hazy_img = batch['source'].cuda()
        clear_img = batch['target'].cuda()
        with torch.no_grad():
            output = model(hazy_img)
            loss_s = criterion(output, clear_img)
            clear_fft = calculate_FFT(clear_img)
            output_fft = calculate_FFT(output)
            loss_f = criterion(output_fft, clear_fft)
            loss = loss_s + 0.01 * loss_f

        losses += loss.item() * hazy_img.size(0)

        # [-1, 1] to [0, 1]
        output = output * 0.5 + 0.5
        clear_img = clear_img * 0.5 + 0.5

        running_psnr = 10 * torch.log10(1 / F.mse_loss(output, clear_img)).item()
        total_psnr += running_psnr

        _, _, H, W = output.size()
        down_ratio = max(1, round(min(H, W) / 256))

        count_ssim = 0.0
        count = 0
        for i in range(output.shape[0]):
            running_ssim = ssim(F.adaptive_avg_pool2d(output[i:i + 1], (int(H / down_ratio), int(W / down_ratio))),
                                F.adaptive_avg_pool2d(clear_img[i:i + 1], (int(H / down_ratio), int(W / down_ratio))),
                                data_range=1, size_average=False).item()
            count += 1
            count_ssim += running_ssim
        avg_ssim = count_ssim / count
        total_ssim += avg_ssim

    return losses, total_psnr, total_ssim


if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Train on CPU ...', flush=True)
    else:
        print('CUDA is available. Train on GPU ...', flush=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model()
    model.to(device)
    epochs = 500
    jundge_autocast = True
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001 * 1e-2)
    criterion = torch.nn.L1Loss()

    transform = transforms.Compose([transforms.ToTensor()])
    target_size = (256, 256)

    train_clear_images, train_hazy_images = get_dir('train')
    valid_clear_images, valid_hazy_images = get_dir('valid')
    test_clear_images, test_hazy_images = get_dir('test')

    train_dataset = Loader(train_clear_images, train_hazy_images, 'train')
    valid_dataset = Loader(valid_clear_images, valid_hazy_images, 'valid')
    test_dataset = Loader(test_clear_images, test_hazy_images, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    save_dir = './'
    best_losses = 100
    for epoch in range(epochs):
        now = datetime.now()
        print("current time:", now, flush=True)
        print('Epoch:{}/{}'.format(epoch + 1, epochs), flush=True)
        print('=' * 20, flush=True)
        for phase in ['train', 'valid', 'test']:
            if phase == 'train':
                losses, total_psnr, total_ssim = train(train_dataloader, model, criterion, optimizer, jundge_autocast,
                                                       scaler)
                scheduler.step()
                losses = losses / len(train_dataloader)
                total_psnr = total_psnr / len(train_dataloader)
                total_ssim = total_ssim / len(train_dataloader)
            elif phase == 'valid':
                losses, total_psnr, total_ssim = valid(valid_dataloader, model, criterion)
                losses = losses / len(valid_dataloader)
                total_psnr = total_psnr / len(valid_dataloader)
                total_ssim = total_ssim / len(valid_dataloader)
                if losses < best_losses:
                    best_losses = losses
                    print('save_model,loss:', best_losses)
                    torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'loss': best_losses,
                        'psnr': total_psnr,
                        'ssim': total_ssim,
                        'epoch': epoch,
                    },
                        os.path.join(save_dir, 'best.pth')
                    )
            elif phase == 'test':
                losses, total_psnr, total_ssim = test(test_dataloader, model, criterion)
                lens = len(test_dataloader)
                losses = losses / len(test_dataloader)
                total_psnr = total_psnr / len(test_dataloader)
                total_ssim = total_ssim / len(test_dataloader)

            print('{} Loss: {:.4f}'.format(phase, losses), flush=True)
            print('{} PSNR: {:.4f}'.format(phase, total_psnr), flush=True)
            print('{} SSIM: {:.4f}'.format(phase, total_ssim), flush=True)
