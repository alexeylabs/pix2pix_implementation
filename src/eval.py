from config import *
from model.discriminator import Discriminator
from model.generator import Generator
import torch
from torch import nn
from data.dataset import ImageDataset
from torch.utils.data import DataLoader
import time
from train import train
from utils.save_generated_images import save_generated_images

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    print('Starting training. Device is ' + DEVICE)

    discriminator = Discriminator(in_channels=6).to(DEVICE)
    generator = Generator().to(DEVICE)

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
                                               lr=LEARNING_RATE_DISCRIMINATOR,
                                               betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(),
                                           lr=LEARNING_RATE_GENERATOR,
                                           betas=(0.5, 0.999))

    criterion_discriminator = nn.BCELoss()
    criterion_generator = nn.L1Loss()

    train_dataset = ImageDataset(TRAIN_IMAGES_PATH)
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_dataset = ImageDataset(VALID_IMAGES_PATH,
                                 use_augmentation=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=5,
                              shuffle=False)

    for epoch in range(NUM_EPOCHS):

        start_time = time.time()
        print('[{0:0=4d}]'.format(epoch), end=' ')
        train(discriminator, generator,
              optimizer_discriminator, optimizer_generator,
              criterion_discriminator, criterion_generator,
              train_loader)
        finish_time = time.time()
        print('Time: {} s'.format(int(finish_time - start_time)))

        generator.eval()
        with torch.no_grad():
            for img, _ in valid_loader:
                generated = generator(img.to(DEVICE)).detach()
                save_generated_images(epoch, img.cpu(), generated.cpu())
                break

        torch.save(generator, RESULTS_PATH+'generator_{0:0=4d}.pt'.format(epoch))


main()
