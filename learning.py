import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
import cv2
from tqdm.notebook import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import custom
import model

DATA_PATH = '/home/lopharb/Документы/datasets/monet2photo'

# transforms
img_width, img_height = 256, 256

transf = torch.Compose(
    # making images a bit larger than the desires size so we can then use random crop
    transforms.Resize((img_width * 1.12, img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
)

# losses
criterion_gen = None


class Lambda_scheduler():
    """
    Parameters:
    * epochs_amount is the amount of epoch in a whole learning session

    * decay_start is the amount of epochs after which the scheduler will start working

    * offset (optional, 0 by default) - amount of epochs, that have passed before"""

    def __init__(self, epochs_amount, decay_start, offset=0):
        self.offset = offset
        self.epchs_amount = epochs_amount
        self.decay_start = decay_start

    def step(self):
        pass


n_cpu = 2  # (or 6 if we're learning locally)
batch_size = 1


# models, optimizers and schedulers
gen_ab = model.Generator()
gen_ba = model.Generator()
dis_a = model.Discriminator()
dis_b = model.Discriminator()

optimizers = {
    'gen_ab': torch.optim.Adam(gen_ab.parameters()),
    'gen_ba': torch.optim.Adam(gen_ba.parameters()),
    'dis_a': torch.optim.Adam(dis_a.parameters()),
    'dis_b': torch.optim.Adam(dis_b.parameters())
}
