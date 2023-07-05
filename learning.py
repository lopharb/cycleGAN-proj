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

# meta
DATA_PATH = '/home/lopharb/Документы/datasets/monet2photo'
n_cpu = 2  # (or 6 if we're learning locally)
batch_size = 1
img_width, img_height = 256, 256

# transforms
transf = torch.Compose(
    # making images a bit larger than the desired size so we can then use random crop
    transforms.Resize(
        (int(img_width * 1.15), int(img_height * 1.15)), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
)

# loading data
# TODO data loaders

# losses
criterion_gan = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()


class Lambda_scheduler():
    """
    Parameters:
    * epochs_amount - the amount of epoch in a whole learning session

    * decay_start - the amount of epochs after which the scheduler will start working

    * offset (optional, 0 by default) - amount of epochs, that have passed in the previous session
    """

    def __init__(self, epochs_amount, decay_start, offset=0):
        self.offset = offset
        self.epchs_amount = epochs_amount
        self.decay_start = decay_start

    def step(self, epoch):
        return 1.0 - max(0, epoch+self.offset - self.decay_start)/(self.epchs_amount - self.decay_start)


# models, optimizers and schedulers
gen_ab = model.Generator()
gen_ba = model.Generator()
dis_a = model.Discriminator()
dis_b = model.Discriminator()

lr = 0.0005
b1 = 0.5
b2 = 0.999
optimizers = {
    'gen_ab': torch.optim.Adam(gen_ab.parameters(), lr, (b1, b2)),
    'gen_ba': torch.optim.Adam(gen_ba.parameters(), lr, (b1, b2)),
    'dis_a': torch.optim.Adam(dis_a.parameters(), lr, (b1, b2)),
    'dis_b': torch.optim.Adam(dis_b.parameters(), lr, (b1, b2))
}

epochs = 200
decay_start = 100
offset = 0
schedulers = {
    'gen_ab': torch.optim.lr_scheduler.LambdaLR(optimizers['gen_ab'],
                                                lr_lambda=Lambda_scheduler(epochs, decay_start)),
    'gen_ba': torch.optim.lr_scheduler.LambdaLR(optimizers['gen_ba'],
                                                lr_lambda=Lambda_scheduler(epochs, decay_start)),
    'dis_a': torch.optim.lr_scheduler.LambdaLR(optimizers['dis_a'],
                                               lr_lambda=Lambda_scheduler(epochs, decay_start)),
    'dis_b': torch.optim.lr_scheduler.LambdaLR(optimizers['dis_b'],
                                               lr_lambda=Lambda_scheduler(epochs, decay_start)),
}
