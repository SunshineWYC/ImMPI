import torch
import numpy as np
import torch.nn.functional as F
from math import exp
import lpips


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


def calMAE(input, target):
    """
    Calculate MAE error
    Args:
        input: estimated result, type:torch.Tensor, shape;[B, C, H, W]
        target: groundtruth result, type:torch.Tensor, shape;[B, C, H, W]

    Returns:
        MAE metrics
    """
    errors = torch.abs(input - target)
    MAE = torch.sum(errors) / (input.shape[0] * input.shape[2] * input.shape[3])
    return MAE


def calSSIM(ssim_calculator, input, target, mask=None):
    """
    Calculate SSIM metric
    Args:
        ssim_calculator: SSIM object instance, type:object-SSIM
        input: estimated result, type:torch.Tensor, shape;[B, C, H, W]
        target: groundtruth result, type:torch.Tensor, shape;[B, C, H, W]
        mask: mask, type:torch.Tensor, shape:[B, 1, H, W], value:0/1


    Returns:
        SSIM metrics

    """
    if mask is not None:
        mask = mask.repeat(1, input.shape[1], 1, 1)
        input = input * mask
        target = target * mask
    with torch.no_grad():
        ssim = ssim_calculator(input, target)
    return ssim


def calPSNR(input, target, mask=None):
    """
    Calculate PSNR metric
    Args:
        input: estimated result, type:torch.Tensor, shape;[B, C, H, W]
        target: groundtruth result, type:torch.Tensor, shape;[B, C, H, W]
        mask: mask, type:torch.Tensor, shape:[B, 1, H, W], value:0/1

    Returns:
        PSNR metrics
    """
    if mask is not None:
        mask = mask.repeat(1, input.shape[1], 1, 1)
        input = input * mask
        target = target * mask
        mse = torch.sum((input - target) ** 2, dim=(1, 2, 3)) / torch.sum(mask, dim=(1, 2, 3))
    else:
        mse = ((input - target) ** 2).mean((1, 2, 3))
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # [B, ]

    return psnr.mean()


def calLPIPS(lpips_calculator, input, target, mask=None):
    """
    Calculate LPIPS metric
    Args:
        lpips_calculator: lpips.LPIPS object instance, type:object-LPIPS
        input: estimated result, type:torch.Tensor, shape;[B, C, H, W]
        target: groundtruth result, type:torch.Tensor, shape;[B, C, H, W]
        mask: mask, type:torch.Tensor, shape:[B, 1, H, W], value:0/1

    Returns:
        LPIPS metrics
    """
    if mask is not None:
        mask = mask.repeat(1, input.shape[1], 1, 1)
        input = input * mask
        target = target * mask
    with torch.no_grad():
        lpips_value = lpips_calculator(input, target)
    return lpips_value.mean()


def calDepthmapAccuracy(depth_estimated, depth_ref, thresh, mask=None):
    """
    Compute 2D depth map accuracy
    Args:
        depth_estimated: depth_estimated by network, type:torch.Tensor, shape:[B, 1, H, W]
        depth_ref: depth ground truth from dataloader, type:torch.Tensor,bool, shape:[B, 1, H, W]
        thresh: distance under thresh considered as accurate estimate, type:Union[float, int]
        mask: if is not None, all True, type:torch.Tensor, shape:[B, 1, H, W]

    Returns: accurate_rate, type:float

    """
    if mask is None:
        mask = torch.ones_like(depth_estimated, device=depth_estimated.device).bool()
    depth_estimated = depth_estimated[mask]
    depth_gt = depth_ref[mask]
    errors = torch.abs(depth_gt - depth_estimated)
    error_mask = errors < thresh
    accurate_rate = torch.sum(error_mask.float()) / torch.sum(mask.float())

    return accurate_rate


if __name__ == '__main__':
    batch_size, height, width = 1, 128, 128
    input = torch.ones((batch_size, 3, height, width), dtype=torch.float32) * 0.99
    target = torch.ones((batch_size, 3, height, width), dtype=torch.float32)

    mask = torch.ones((batch_size, 1, height, width), dtype=torch.float32)

    psnr = calPSNR(input, target, mask)
    print(psnr)

    # ssim_calculator = SSIM()
    # ssim_value = calSSIM(ssim_calculator, input, target, mask)
    # print(ssim_value.item())

    lpips_calculator = lpips.LPIPS(net="vgg")
    lpips_value = calLPIPS(lpips_calculator, input, target, mask)
    print(lpips_value.item())
