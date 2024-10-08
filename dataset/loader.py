import cv2
import numpy as np
from torch.utils.data import Dataset
import augment

def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs


def read_img(filename):
    img = cv2.imread(filename)
    return img[:, :, ::-1].astype('float32') / 255.0


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()


def get_dir(name):
    hazy_img = []
    clear_img = []
    if name == 'train':
        with open('./dataset/train_haze4k_shuffle.txt', 'r') as f:
            for line in f:
                line = line.split()
                clear_img.append(line[1])
                hazy_img.append(line[2])
    elif name == 'valid':
        with open('./dataset/valid_haze4k_shuffle.txt', 'r') as f:
            for line in f:
                line = line.split()
                clear_img.append(line[1])
                hazy_img.append(line[2])
    elif name == 'test':
        with open('./dataset/test_haze4k_shuffle.txt', 'r') as f:
            for line in f:
                line = line.split()
                clear_img.append(line[1])
                hazy_img.append(line[2])

    return clear_img, hazy_img


class Loader(Dataset):
    def __init__(self, clear_images, hazy_images, mode, size=256, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.clear_images = clear_images
        self.hazy_images = hazy_images

    def __len__(self):
        return len(self.clear_images)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        clear_images = read_img(self.clear_images[idx]) * 2 - 1
        hazy_images = read_img(self.hazy_images[idx]) * 2 - 1

        if self.mode == 'train':
            [hazy_images, clear_images] = augment([hazy_images, clear_images], self.size, self.edge_decay,
                                                  self.only_h_flip)

        if self.mode == 'valid':
            [hazy_images, clear_images] = align([hazy_images, clear_images], self.size)

        return {'source': hwc_to_chw(hazy_images), 'target': hwc_to_chw(clear_images)}
