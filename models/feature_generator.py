import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class FeatureGenerator(nn.Module):
    def __init__(self, model_type, pretrained=True, device=torch.device("cuda")):
        super(FeatureGenerator, self).__init__()

        assert model_type in ["resnet18", "resnet50"]
        resnets_dict = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50
        }
        encoder_channels = {
            "resnet18": np.array([64, 64, 128, 256, 512]),
            "resnet50": np.array([64, 256, 512, 1024, 2048]),
        }

        self.encoder = resnets_dict[model_type](pretrained=pretrained)
        self.encoder_channels = encoder_channels[model_type]

        # image normalize parameters
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)
        self.img_mean = self.img_mean.view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)
        self.img_std = self.img_std.view(1, 3, 1, 1)

    def forward(self, ref_images):
        """

        Args:
            ref_images: input image, type:torch.Tensor, shape:[B, 3, H, W]

        Returns:
            conv1_out: type:torch.Tensor, shape:[B, C, H/2, W/2]
            block1_out: type:torch.Tensor, shape:[B, C, H/4, W/4]
            block2_out: type:torch.Tensor, shape:[B, C, H/8, W/8]
            block3_out: type:torch.Tensor, shape:[B, C, H/16, W/16]
            block4_out: type:torch.Tensor, shape:[B, C, H/32, W/32]
        """
        ref_images_normalized = (ref_images - self.img_mean) / self.img_std

        conv1_out = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(ref_images_normalized)))
        block1_out = self.encoder.layer1(self.encoder.maxpool(conv1_out))
        block2_out = self.encoder.layer2(block1_out)
        block3_out = self.encoder.layer3(block2_out)
        block4_out = self.encoder.layer4(block3_out)

        return conv1_out, block1_out, block2_out, block3_out, block4_out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureGenerator(model_type="resnet18", pretrained=True, device=device).to(device)

    batch_size, height, width = 2, 256, 256
    ref_images = 256 * torch.rand((batch_size, 3, height, width), device=device)
    conv1_out, block1_out, block2_out, block3_out, block4_out = model(ref_images)
    print("conv1_out", conv1_out.shape)
    print("block1_out", block1_out.shape)
    print("block2_out", block2_out.shape)
    print("block3_out", block3_out.shape)
    print("block4_out", block4_out.shape)

    total_params = sum(params.numel() for params in model.parameters())
    train_params = sum(params.numel() for params in model.parameters() if params.requires_grad)
    print("total_paramteters: {}, train_parameters: {}".format(total_params, train_params))

