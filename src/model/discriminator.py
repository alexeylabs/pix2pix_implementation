import torch
import torch.nn as nn
from model.blocks import EncodeBlock

# 6.1.2 Discriminator architectures
# The 70 × 70 discriminator architecture is:
# C64-C128-C256-C512
# After the last layer, a convolution is applied to map to
# a 1-dimensional output, followed by a Sigmoid function.
# As an exception to the above notation, BatchNorm is not
# applied to the first C64 layer. All ReLUs are leaky, with
# slope 0.2.


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64,
                      kernel_size=4, stride=2,
                      padding=1),
            nn.LeakyReLU(0.2),
            )
        self.model = nn.Sequential(
            EncodeBlock(64, 128),
            EncodeBlock(128, 256),
            EncodeBlock(256, 512),
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(512, 1,
                      kernel_size=4, stride=1,
                      padding=1),
            nn.Sigmoid(),
            )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.first_layer(x)
        x = self.model(x)
        x = self.last_layer(x)
        return x
