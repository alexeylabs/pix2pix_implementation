import torch
from config import *
import numpy as np


def train(discriminator, generator,
          optimizer_discriminator, optimizer_generator,
          criterion_discriminator, criterion_generator,
          train_loader, show_scores=True):
    real_score_per_epoch = []
    fake_score_per_epoch = []

    loss_d_per_epoch = []
    loss_g_per_epoch = []

    for images, targets in train_loader:
        discriminator.train()
        generator.train()

        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Train discriminator
        optimizer_discriminator.zero_grad()

        fake_targets = generator(images)
        real_preds = discriminator(images, targets)
        fake_preds = discriminator(images, fake_targets)
        real_loss = criterion_discriminator(real_preds, torch.ones_like(real_preds))
        fake_loss = criterion_discriminator(fake_preds, torch.zeros_like(fake_preds))
        discriminator_loss = real_loss + fake_loss

        real_score_per_epoch.append(torch.mean(real_preds).item())
        fake_score_per_epoch.append(torch.mean(fake_preds).item())

        discriminator_loss.backward()
        optimizer_discriminator.step()

        # Train generator
        optimizer_generator.zero_grad()

        fake_targets = generator(images)
        fake_preds = discriminator(images, fake_targets)
        fool_loss = criterion_discriminator(fake_preds, torch.ones_like(fake_preds))
        l1_loss = criterion_generator(fake_targets,
                                      targets) * 100
        generator_loss = fool_loss + l1_loss  # Adding both terms together (with λ = 100) reduces these artifacts

        generator_loss.backward()
        optimizer_generator.step()

        loss_d_per_epoch.append(discriminator_loss.item())
        loss_g_per_epoch.append(generator_loss.item())

    if show_scores:
        print(f'Real score: {np.mean(real_score_per_epoch)}\t Fake score: {np.mean(fake_score_per_epoch)}')
        print(f'Discriminator loss: {np.mean(loss_d_per_epoch)}\t Generator loss: {np.mean(loss_g_per_epoch)}')
