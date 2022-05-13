"""
Loss Functions
"""

import torch
import torch.nn.functional as F
from math import exp


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


def loss_fcn_rgb_SSIM(ssim_calculator, tgt_rgb_syn, tgt_mask, image_tgt):
    """
    Calculate SSIM loss between rgb syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        ssim_calculator: SSIM object instance, type:object-SSIM
        tgt_rgb_syn: tgt synthetic rgb images, type:torch.Tensor, shape:[B, 3, H, W]
        tgt_mask: tgt synthetic masks, type:torch.Tensor, shape:[B, 1, H, W], value:0or1
        image_tgt: tgt groundtruth rgb images, type:torch.Tensor, shape:[B, 3, H, W]

    Returns:
        loss_ssim: ssim loss between rgb_syn and rgb_gts

    """
    loss_ssim = ssim_calculator(tgt_rgb_syn * tgt_mask, image_tgt * tgt_mask)
    return 1 - loss_ssim


def loss_fcn_rgb_lpips(lpips_calculator, tgt_rgb_syn, tgt_mask, image_tgt):
    """
    Calculate LPIPS loss between rgb syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        lpips_calculator: lpips.LPIPS object instance, type:object-LPIPS
        tgt_rgb_syn: tgt synthetic rgb images, type:torch.Tensor, shape:[B, 3, H, W]
        tgt_mask: tgt synthetic masks, type:torch.Tensor, shape:[B, 1, H, W], value:0or1
        image_tgt: tgt groundtruth rgb image, type:torch.Tensor, shape:[B, 3, H, W]

    Returns:
        loss_lpips: loss between rgb_syn and rgb_gts
    """
    loss_lpips = lpips_calculator(tgt_rgb_syn * tgt_mask, image_tgt * tgt_mask).mean()
    return loss_lpips


def loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, image_tgt):
    """
    Calculate smooth-L1 loss between rgb syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        tgt_rgb_syn: tgt synthetic rgb images, type:torch.Tensor, shape:[B, 3, H, W]
        tgt_mask: tgt synthetic masks, type:torch.Tensor, shape:[B, 1, H, W], value:0or1
        image_tgt: tgt groundtruth rgb image, type:torch.Tensor, shape:[B, 3, H, W]

    Returns:
        loss_rgb: loss between rgb_syn and rgb_gts

    """
    # calculate tgt-view and ref-view L1 rgb loss, with mask
    loss_rgb = torch.sum(torch.abs(tgt_rgb_syn * tgt_mask - image_tgt * tgt_mask)) / torch.sum(tgt_mask)
    # loss_rgb = F.l1_loss(tgt_rgb_syn * tgt_mask, image_tgt * tgt_mask, reduction="mean")
    return loss_rgb


def loss_fcn_rgb_L2(tgt_rgb_syn, tgt_mask, image_tgt):
    """
    Calculate smooth-L1 loss between rgb syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        tgt_rgb_syn: tgt synthetic rgb images, type:torch.Tensor, shape:[B, 3, H, W]
        tgt_mask: tgt synthetic masks, type:torch.Tensor, shape:[B, 1, H, W], value:0or1
        image_tgt: tgt groundtruth rgb image, type:torch.Tensor, shape:[B, 3, H, W]

    Returns:
        loss_rgb: loss between rgb_syn and rgb_gts

    """
    # calculate tgt-view and ref-view L1 rgb loss, with mask
    loss_rgb = torch.sum((tgt_rgb_syn * tgt_mask - image_tgt * tgt_mask) ** 2) / torch.sum(tgt_mask)
    # loss_rgb = F.l1_loss(tgt_rgb_syn * tgt_mask, image_tgt * tgt_mask, reduction="mean")
    return loss_rgb


def loss_fcn_edge_aware(ref_depth_syn, image_ref, depth_min_ref, depth_max_ref):
    """
    Calculate edge-aware loss between depth syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        ref_depth_syn: ref synthetic depth, type:torch.Tensor, shape:[B, 1, H, W]
        image_ref: ref-view groundtruth rgb image, type:torch.Tensor, shape:[B, 3, H, W]
        depth_min_ref: depth min value, type:torch.Tensor, shape:[B,]
        depth_max_ref: depth max value, type:torch.Tensor, shape:[B,]

    Returns:
        loss_edge: loss between depth syn and rgb groundtruth
    """
    depth_min_ref = depth_min_ref.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, ref_depth_syn.shape[2], ref_depth_syn.shape[3])
    depth_max_ref = depth_max_ref.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, ref_depth_syn.shape[2], ref_depth_syn.shape[3])
    ref_depth_syn = (ref_depth_syn - depth_min_ref) / (depth_max_ref - depth_min_ref)

    # calculate depth gradient
    grad_depth_x = torch.abs(ref_depth_syn[:, :, :, :-1] - ref_depth_syn[:, :, :, 1:])  # [B, 1, H, W-1]
    grad_depth_y = torch.abs(ref_depth_syn[:, :, :-1, :] - ref_depth_syn[:, :, 1:, :])  # [B, 1, H, W-1]
    # calculate image gradient
    grad_image_x = torch.mean(torch.abs(image_ref[:, :, :, :-1] - image_ref[:, :, :, 1:]), 1, keepdim=True) # [B, 1, H, W-1]
    grad_image_y = torch.mean(torch.abs(image_ref[:, :, :-1, :] - image_ref[:, :, 1:, :]), 1, keepdim=True) # [B, 1, H, W-1]

    loss_edge = torch.mean(grad_depth_x * torch.exp(-grad_image_x)) + torch.mean(grad_depth_y * torch.exp(-grad_image_y))
    return loss_edge


if __name__ == '__main__':
    batch_size, height, width = 2, 512, 512
    neighbor_view_num = 4
    tgt_rgb_syn = torch.ones((batch_size, 3, height, width), dtype=torch.float32) * 0.2
    tgt_rgb_syns = [tgt_rgb_syn for i in range(neighbor_view_num)]
    tgt_mask = torch.ones((batch_size, 3, height, width), dtype=torch.float32)
    tgt_masks = [tgt_mask for i in range(neighbor_view_num)]
    images_tgt = torch.ones((batch_size, neighbor_view_num, 3, height, width), dtype=torch.float32)

    # For 4neighbor training
    loss_rgb_l1 = loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, images_tgt[:, 1, :, :, :])
    print(loss_rgb_l1.item())
    
    ssim_calculator = SSIM()
    loss_ssim = loss_fcn_rgb_SSIM(ssim_calculator, tgt_rgb_syn, tgt_mask, images_tgt[:, 1, :, :, :])
    print(loss_ssim.item())

    # lpips_calculator = lpips.LPIPS(net="vgg")
    # lpips_calculator.requires_grad = False
    # loss_lpips = loss_fcn_rgb_lpips(lpips_calculator, tgt_rgb_syns, tgt_masks, images_tgt)
    # print(loss_lpips.item())
    # loss_lpips = loss_fcn_rgb_lpips(lpips_calculator, tgt_rgb_syns, tgt_masks, images_tgt)
    # print(loss_lpips.item())

    # For general use
    # loss_rgb_l1 = loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, images_tgt[:, 0, :, :, :])
    # loss_rgb_smooth_l1 = loss_fcn_rgb_Smooth_L1(tgt_rgb_syn, tgt_mask, images_tgt[:, 0, :, :, :])
    # print(loss_rgb_l1.item(), loss_rgb_smooth_l1.item())
    #
    # ssim_calculator = SSIM()
    # loss_ssim = loss_fcn_rgb_SSIM(ssim_calculator, tgt_rgb_syn, tgt_mask, images_tgt[:, 0, :, :, :])
    # print(loss_ssim.item())
    #
    # lpips_calculator = lpips.LPIPS(net="vgg")
    # lpips_calculator.requires_grad = False
    # loss_lpips = loss_fcn_rgb_lpips(lpips_calculator, tgt_rgb_syn, tgt_mask, images_tgt[:, 0, :, :, :])
    # print(loss_lpips.item())

    # import cv2
    # depth_min_ref = torch.tensor([485.703,], dtype=torch.float32)
    # depth_max_ref = torch.tensor([536.844,], dtype=torch.float32)
    # ref_depth_syn = torch.from_numpy(cv2.imread("../testdata/depth_007.png", cv2.IMREAD_ANYDEPTH) / 64.0).unsqueeze(0).unsqueeze(1)
    # ref_depth_syn = ref_depth_syn + torch.randn(ref_depth_syn.shape) * 0.

    # image_ref = torch.from_numpy(cv2.imread("../testdata/image_007.png") / 255.0).unsqueeze(0).permute(0, 3, 1, 2)
    # loss_edge = loss_fcn_edge_aware(ref_depth_syn, image_ref, depth_min_ref, depth_max_ref)
    # print(loss_edge)
