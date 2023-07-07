import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from glob import glob
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import custom
from custom import Lambda_scheduler
import model

# meta
DATA_PATH = '/home/lopharb/Документы/datasets/monet2photo'
n_cpu = 6  # (or 6 if we're learning locally)
batch_size = 1
img_width, img_height = 256, 256

# transforms
transf = transforms.Compose(
    # making images a bit larger than the desired size so we can then use random crop
    [transforms.Resize(
        (int(img_width * 1.15), int(img_height * 1.15)), Image.BICUBIC),
     transforms.RandomCrop((img_height, img_width)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# loading data
class ImageDataset(Dataset):
    # a is monet, b is photo
    def __init__(self, root: str, transforms,  mode: str) -> None:
        self.mode = mode
        self.transforms = transforms
        if mode == 'train':
            self.files_a = sorted(glob(os.path.join(root+'/trainA')+'/*.*'))
            self.files_b = sorted(
                glob(os.path.join(root+'/trainB')+'/*.*')[:len(self.files_a)])  # slicing so the datasets are matched by size
        if mode == 'test':
            self.files_a = sorted(glob(os.path.join(root+'/testA')+'/*.*'))
            self.files_b = sorted(
                glob(os.path.join(root+'/testB')+'/*.*')[:len(self.files_a)])

    def __getitem__(self, index):
        img_a = Image.open(self.files_a[index % len(self.files_a)])
        img_b = Image.open(self.files_b[index % len(self.files_b)])
        if img_a.mode != 'RGB':
            img_a = custom.to_rgb(img_a)
        if img_b.mode != 'RGB':
            img_b = custom.to_rgb(img_b)

        item_a = self.transforms(img_a)
        item_b = self.transforms(img_b)
        return {'a': item_a, 'b': item_b}

    def __len__(self):
        return len(self.files_a)


train_loader = DataLoader(
    ImageDataset(DATA_PATH, transf, 'train'),
    batch_size=1,
    shuffle=True,
    num_workers=n_cpu
)

test_loader = DataLoader(
    ImageDataset(DATA_PATH, transf, 'test'),
    batch_size=1,
    shuffle=True,
    num_workers=n_cpu
)

# losses
criterion_gan = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# models, optimizers and schedulers
gen_ab = model.Generator(5).to('cuda')
gen_ba = model.Generator(5).to('cuda')
dis_a = model.Discriminator().to('cuda')
dis_b = model.Discriminator().to('cuda')

lr = 0.0005
b1 = 0.5
b2 = 0.999
optimizers = {
    'gen_ab': torch.optim.Adam(gen_ab.parameters(), lr, (b1, b2)),
    'gen_ba': torch.optim.Adam(gen_ba.parameters(), lr, (b1, b2)),
    'dis_a': torch.optim.Adam(dis_a.parameters(), lr, (b1, b2)),
    'dis_b': torch.optim.Adam(dis_b.parameters(), lr, (b1, b2))
}

epochs = 100
decay_start = 50
offset = 0
schedulers = {
    'gen_ab': torch.optim.lr_scheduler.LambdaLR(optimizers['gen_ab'],
                                                lr_lambda=Lambda_scheduler(epochs, decay_start).step),
    'gen_ba': torch.optim.lr_scheduler.LambdaLR(optimizers['gen_ba'],
                                                lr_lambda=Lambda_scheduler(epochs, decay_start).step),
    'dis_a': torch.optim.lr_scheduler.LambdaLR(optimizers['dis_a'],
                                               lr_lambda=Lambda_scheduler(epochs, decay_start).step),
    'dis_b': torch.optim.lr_scheduler.LambdaLR(optimizers['dis_b'],
                                               lr_lambda=Lambda_scheduler(epochs, decay_start).step),
}

# creating buffers for discriminators
img_a_buffer = custom.Buffer(50)
labels_a_buffer = custom.Buffer(50)
img_b_buffer = custom.Buffer(50)
labels_b_buffer = custom.Buffer(50)

real_label = torch.ones((1, 1, 1, 1)).to('cuda')
fake_label = torch.zeros((1, 1, 1, 1)).to('cuda')

# training loop
for epoch in range(epochs):
    # train
    for batch_num, imgs in enumerate(tqdm(train_loader)):
        # preparing data
        real_a = imgs['a'].to('cuda')
        real_b = imgs['b'].to('cuda')
        # TRAINING GENERATORS
        gen_ab.train()
        gen_ba.train()
        optimizers['gen_ab'].zero_grad()
        optimizers['gen_ba'].zero_grad()

        # identity loss
        id_loss_a = criterion_identity(gen_ab(real_b), real_b)
        id_loss_b = criterion_identity(gen_ba(real_a), real_a)

        id_loss = (id_loss_a+id_loss_b)/2

        # gan loss
        fake_a = gen_ba(real_b)
        fake_b = gen_ab(real_a)
        gan_loss_a = criterion_gan(dis_a(fake_a), real_label)
        gan_loss_b = criterion_gan(dis_b(fake_b), real_label)

        gan_loss = (gan_loss_a+gan_loss_b)/2

        # cycle consistency loss
        recovered_a = gen_ba(fake_b)
        recovered_b = gen_ab(fake_a)
        cc_loss_a = criterion_cycle(recovered_a, real_a)
        cc_loss_b = criterion_cycle(recovered_b, real_b)

        cc_loss = (cc_loss_a+cc_loss_b)/2

        # optimizing generators, found suggested loss weights on the internet
        gen_loss = gan_loss + 10*cc_loss + 5*id_loss
        gen_loss.backward()
        optimizers['gen_ab'].step()
        optimizers['gen_ba'].step()

        # TRAINING DISCRIMINATORS
        dis_a.train()
        dis_b.train()
        # discriminator a
        dis_loss_a = criterion_gan(dis_a(real_a), real_label)
        + criterion_gan(dis_a(fake_a), fake_label)
        dis_loss_a.backward()
        optimizers['dis_a'].step()
        optimizers['dis_a'].zero_grad()

        # discriminator b
        dis_loss_b = criterion_gan(dis_b(real_b), real_label)
        + criterion_gan(dis_b(fake_b), fake_label)
        dis_loss_b.backward()
        optimizers['dis_b'].step()
        optimizers['dis_b'].zero_grad()

        dis_loss = (dis_loss_a+dis_loss_b)/2

        if batch_num % 100 == 0:
            custom.sample_images(gen_ab, gen_ba, test_loader)
            print('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f - (adv : %f, cycle : %f, identity : %f)]'
                  % (epoch+1, epochs,       # [Epoch -]
                     batch_num+1, len(train_loader),   # [Batch -]
                     dis_loss.item(),       # [D loss -]
                     gen_loss.item(),       # [G loss -]
                     gan_loss.item(),     # [adv -]
                     cc_loss.item(),   # [cycle -]
                     id_loss.item(),  # [identity -]
                     ))

    for batch_num, imgs in enumerate(tqdm(test_loader)):
        real_a = imgs['a'].to('cuda')
        real_b = imgs['b'].to('cuda')

        gen_ab.valid()
        gen_ba.valid()

        with torch.no_grad():
            id_loss_a = criterion_identity(gen_ab(real_b), real_b)
            id_loss_b = criterion_identity(gen_ba(real_a), real_a)

            id_loss = (id_loss_a+id_loss_b)/2

            # gan loss
            fake_a = gen_ba(real_b)
            fake_b = gen_ab(real_a)
            gan_loss_a = criterion_gan(dis_a(fake_a), real_label)
            gan_loss_b = criterion_gan(dis_b(fake_b), real_label)

            gan_loss = (gan_loss_a+gan_loss_b)/2

            # cycle consistency loss
            recovered_a = gen_ba(fake_b)
            recovered_b = gen_ab(fake_a)
            cc_loss_a = criterion_cycle(recovered_a, real_a)
            cc_loss_b = criterion_cycle(recovered_b, real_b)

            cc_loss = (cc_loss_a+cc_loss_b)/2
            gen_loss = gan_loss + 10*cc_loss + 5*id_loss

        dis_a.valid()
        dis_b.valid()
        with torch.no_grad():
            # discriminator a
            dis_loss_a = criterion_gan(dis_a(real_a), real_label)
            + criterion_gan(dis_a(fake_a), fake_label)

            # discriminator b
            dis_loss_b = criterion_gan(dis_b(real_b), real_label)
            + criterion_gan(dis_b(fake_b), fake_label)

            dis_loss = (dis_loss_a+dis_loss_b)/2

        if batch_num % 100 == 0:
            custom.sample_images(gen_ab, gen_ba, test_loader)
            print('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f - (adv : %f, cycle : %f, identity : %f)]'
                  % (epoch+1, epochs,       # [Epoch -]
                     batch_num+1, len(train_loader),   # [Batch -]
                     dis_loss.item(),       # [D loss -]
                     gen_loss.item(),       # [G loss -]
                     gan_loss.item(),     # [adv -]
                     cc_loss.item(),   # [cycle -]
                     id_loss.item(),  # [identity -]
                     ))
