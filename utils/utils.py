import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image


def print_args(args):
    """
    Utilities to print arguments
    Arsg:
        args: arguments to pring out
    """
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


def dict2cuda(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2cuda(v)
        elif isinstance(v, torch.Tensor):
            v = v.cuda()
        new_dic[k] = v
    return new_dic


def dict2float(data):
    """Convert tensor to float, each tensor is only one element"""
    new_dict = {}
    for key, value in data.items():
        if isinstance(value, float):
            value = value
        elif isinstance(value, torch.Tensor):
            value = value.data.item()
        else:
            raise NotImplementedError("invalid input type {} for dict2float".format(type(value)))
        new_dict[key] = value
    return new_dict


def save_scalars(logger, mode, scalar_outputs, global_step):
    """
    Write scalar dict to tensorboard logger
    Args:
        logger: tb.SummaryWriter
        mode: "train", "test"
        scalar_outputs: dict[str:torch.Tensor]
        global_step: iteration number

    Returns: None
    """
    scalar_dict = dict2float(scalar_outputs)
    for key, value in scalar_dict.items():
        if isinstance(value, float):
            name = "scalar_{}/{}".format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            raise NotImplementedError("invalid input type {} for save_scalars".format(type(value)))


def save_images(logger, mode, image_outputs, global_step):
    """
    Write image dict to tensorboard logger
    Args:
        logger: tb.SummaryWriter
        mode: "train", "test"
        image_outputs: dict[str:torch.Tensor], image shape is [B, H, W] or [B, C, H, W]
        global_step: iteration number

    Returns:

    """

    image_dict = dict2numpy(image_outputs)

    def preprocess(name, img):  # for [B, C, H, W], stitching to grid to show
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img)
        image_grid = vutils.make_grid(img, nrow=2, padding=1, normalize=True, scale_each=True)
        return image_grid

    for key, value in image_dict.items():
        if isinstance(value, np.ndarray):
            name = "image_{}/{}".format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            raise NotImplementedError("invalid input type {} for save_images".format(type(value)))


def dict2numpy(data):
    """Convert tensor to float, each tensor is array"""
    new_dict = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            value = value
        elif isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().copy()
        else:
            raise NotImplementedError("invalid input type {} for dict2numpy".format(type(value)))
        new_dict[key] = value
    return new_dict


class ScalarDictMerge:
    """Merge scalar dict from each iteration, to get epoch average result"""

    def __init__(self):
        self.data = {}  # store sum of scalar per iteration
        self.count = 0  # store current iteration number

    def update(self, new_input):
        """
        Update self.data
        Args:
            new_input: new data to merge, type:dict{str:float}
        """
        self.count += 1
        new_input = dict2float(new_input)
        if len(self.data) == 0:
            for key, value in new_input.items():
                if not isinstance(value, float):
                    raise NotImplementedError("invalid data {}: {}".format(key, type(value)))
                else:
                    self.data[key] = value
        else:
            for key, value in new_input.items():
                if not isinstance(value, float):
                    raise NotImplementedError("invalid data {}: {}".format(key, type(value)))
                else:
                    self.data[key] += value

    def mean(self):
        """
        Compute average value stored in self.data
        Returns: average value dict

        """
        return {key : value/self.count for key, value in self.data.items()}

