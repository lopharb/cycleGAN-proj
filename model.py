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


class Generator(nn.Module):
    def __init__(self, n_blocks) -> None:
        super(Generator, self).__init__()
        in_channels = 3  # since we have an rgb image
        res_channels = 64

        # initial convolution to get to a desired amount of channels for residual blocks
        model = [
            nn.RefectionPad2d(2),
            nn.Conv2d(in_channels, res_channels, 5),
            nn.InstanceNorm2d(res_channels),
            nn.ReLU()
        ]

        # add downsampling

        # adding residual blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(res_channels)]

        # add upsampling

        # adding final layer
        model += [
            nn.RefectionPad2d(2),
            nn.Conv2d(res_channels, in_channels, 5),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)
