import torch


def calculate_FFT(image):
    image_fft = torch.fft.fft2(image, dim=(-2, -1))
    image_fft = torch.stack((image_fft.real, image_fft.imag), -1)
    return image_fft
