import torch
import torch.nn as nn
from model.blocks import EncodeBlock, DecodeBlock

# 6.1.1 Generator architectures
# The encoder-decoder architecture consists of:
# encoder:
# C64-C128-C256-C512-C512-C512-C512-C512
# decoder:
# CD512-CD512-CD512-C512-C256-C128-C64
# After the last layer in the decoder, a convolution is applied to map to the number of output channels (3 in general,
# except in colorization, where it is 2), followed by a Tanh
# function. As an exception to the above notation, BatchNorm is not applied to the first C64 layer in the encoder.
# All ReLUs in the encoder are leaky, with slope 0.2, while
# ReLUs in the decoder are not leaky.
# The U-Net architecture is identical except with skip connections between each layer i in the encoder and layer n−i
# in the decoder, where n is the total number of layers. The
# skip connections concatenate activations from layer i to
# layer n − i. This changes the number of channels in the
# decoder:
# U-Net decoder:
# CD512-CD1024-CD1024-C1024-C1024-C512
# -C256-C128


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.enc_conv_0 = nn.Sequential(
            nn.Conv2d(in_channels, 64,
                      kernel_size=4, stride=2,
                      padding=1),
            nn.LeakyReLU(0.2))
        self.enc_conv_1 = EncodeBlock(64, 128)
        self.enc_conv_2 = EncodeBlock(128, 256)
        self.enc_conv_3 = EncodeBlock(256, 512)
        self.enc_conv_4 = EncodeBlock(512, 512)
        self.enc_conv_5 = EncodeBlock(512, 512)
        self.enc_conv_6 = EncodeBlock(512, 512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512,
                      kernel_size=4, stride=2,
                      padding=1))

        self.dec_conv_7 = DecodeBlock(512, 512, use_dropout=True)
        self.dec_conv_6 = DecodeBlock(512 * 2, 512, use_dropout=True)
        self.dec_conv_5 = DecodeBlock(512 * 2, 512, use_dropout=True)
        self.dec_conv_4 = DecodeBlock(512 * 2, 512)
        self.dec_conv_3 = DecodeBlock(512 * 2, 256)
        self.dec_conv_2 = DecodeBlock(256 * 2, 128)
        self.dec_conv_1 = DecodeBlock(128 * 2, 64)
        self.dec_conv_0 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, in_channels,
                               kernel_size=4, stride=2,
                               padding=1),
            nn.Tanh())

    def forward(self, x):
        e0 = self.enc_conv_0(x)   # 256 - > 128
        e1 = self.enc_conv_1(e0)  # 128 - > 64
        e2 = self.enc_conv_2(e1)  # 64  - > 32
        e3 = self.enc_conv_3(e2)  # 32  - > 16
        e4 = self.enc_conv_4(e3)  # 16  - > 8
        e5 = self.enc_conv_5(e4)  # 8   - > 4
        e6 = self.enc_conv_6(e5)  # 4   - > 2
        b = self.bottleneck(e6)   # 2   - > 1

        d7 = self.dec_conv_7(b)                           # 1   -> 2
        d6 = self.dec_conv_6(torch.cat([d7, e6], dim=1))  # 2   -> 4
        d5 = self.dec_conv_5(torch.cat([d6, e5], dim=1))  # 4   -> 8
        d4 = self.dec_conv_4(torch.cat([d5, e4], dim=1))  # 8   -> 16
        d3 = self.dec_conv_3(torch.cat([d4, e3], dim=1))  # 16  -> 32
        d2 = self.dec_conv_2(torch.cat([d3, e2], dim=1))  # 32  -> 64
        d1 = self.dec_conv_1(torch.cat([d2, e1], dim=1))  # 64  -> 128
        d0 = self.dec_conv_0(torch.cat([d1, e0], dim=1))  # 128 -> 256
        return d0
