'''
Implementation of SSIM loss
Inherited from https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/ssim.html 
'''

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeometry.image import get_gaussian_kernel2d

class SSIM(nn.Module):
    def __init__(self, window_size: int, reduction: str = "none", max_val: float = 1.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.reduction = reduction
        self.window = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))
        self.padding = self.compute_zero_padding(window_size)
        self.C1 = (0.01*self.max_val)**2
        self.C2 = (0.03*self.max_val)**2

    @staticmethod
    def compute_zero_padding(kernel_size):
        return (kernel_size-1)//2
    
    def filter2D(self, input, kernel, channel):
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)
    
    def forward(self, img1, img2):
        b, c, h, w = img1.shape
        tmp_kernel = self.window.to(img1.device).to(img1.dtype)
        kernel = tmp_kernel.repeat(c, 1, 1, 1)

        # compute local mean per channel
        mu1: torch.Tensor = self.filter2D(img1, kernel, c)
        mu2: torch.Tensor = self.filter2D(img2, kernel, c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        return loss


if __name__ == "__main__":
    ssim_loss = SSIM(window_size=11, reduction="mean")
    img1 = torch.randn((1, 3, 16, 16))
    # img2 = torch.randn_like(img1)
    img2 = torch.randn_like(img1)
    print(ssim_loss(img1, img2))