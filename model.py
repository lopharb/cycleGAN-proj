import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import cv2
from tqdm.notebook import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(channels, channels, 5),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(channels, channels, 5),
            nn.InstanceNorm2d(channels)
            # there should maybe be a relu
        )

    def forward(self, x):
        return self.block(x) + x


class Custom(nn.Module):
    def __init__(self,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        print(x.shape)
        return x


class Generator(nn.Module):
    def __init__(self, n_blocks=9) -> None:
        super(Generator, self).__init__()

        initial_channels = in_channels = 3  # since we have an rgb image
        res_channels = 32

        # initial convolution to get to a desired amount of channels for residual blocks
        net = [
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels, res_channels, 5),
            nn.InstanceNorm2d(res_channels),
            nn.ReLU(inplace=True)
        ]

        # downsampling (by factor of 4)
        in_channels = res_channels
        for _ in range(2):
            res_channels *= 2
            net += [
                nn.Conv2d(in_channels, res_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(res_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = res_channels

        # adding residual blocks
        for _ in range(n_blocks):
            net += [ResidualBlock(res_channels)]

        # upsampling (also by factor of 4)
        for _ in range(2):
            res_channels //= 2
            net += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, res_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_channels = res_channels

        # final layer
        net += [
            nn.ReflectionPad2d(2),
            nn.Conv2d(res_channels, initial_channels, 5),
            nn.Tanh()
        ]

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True) -> None:
        super(DiscBlock, self).__init__()
        block = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if normalize:
            block.append(nn.InstanceNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        in_channels = 3  # still taking rgb image as input

        self.net = nn.Sequential(
            DiscBlock(in_channels, 32, normalize=False),
            Custom(),  # 32 128 128
            DiscBlock(32, 64),
            Custom(),  # 64 64 64
            DiscBlock(64, 128),
            Custom(),  # 128 32 32
            DiscBlock(128, 256),
            Custom(),  # 256 16 16
            DiscBlock(256, 512),
            Custom(),  # 512 8 8
            nn.Conv2d(512, 1, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


dis = Discriminator()
x = torch.randn((1, 3, 256, 256))
print(dis(x).item())
