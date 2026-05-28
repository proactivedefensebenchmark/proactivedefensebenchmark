import torch
import torchvision.transforms as T
from torchvision.transforms import v2

def horizontal_flip(p=1.0):
    return T.RandomHorizontalFlip(p=p)

def vertical_flip(p=1.0):
    return T.RandomVerticalFlip(p=p)

def random_rotation(degrees=(-15, 15)):
    return T.RandomRotation(degrees=degrees)

def random_crop(size=(256,256)):
    return T.RandomCrop(size=size)

def resize(size=(256,256)):
    return T.Resize(size=size)

def gaussian_blur(kernel_size=5, sigma=1.0):
    return T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

def box_blur(kernel_size=5):
    return T.BoxBlur(kernel_size=kernel_size)

def jpeg_compression_transform(quality=80):
    return v2.JPEG(quality=quality)

def salt_and_pepper_noise(x: torch.Tensor, amount: float = 0.05, salt_vs_pepper: float = 0.5):
    assert x.dim() == 4, "Input must be (B,C,H,W)"
    
    B, C, H, W = x.shape
    device = x.device

    rnd = torch.rand(B, 1, H, W, device=device)

    salt_mask = rnd < (amount * salt_vs_pepper)
    pepper_mask = (rnd >= (amount * salt_vs_pepper)) & (rnd < amount)

    x_noisy = x.clone()
    x_noisy[salt_mask.expand(-1, C, -1, -1)] = 1.0
    x_noisy[pepper_mask.expand(-1, C, -1, -1)] = 0.0

    return x_noisy