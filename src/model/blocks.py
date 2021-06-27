import torch.nn as nn

# 6.1. Network architectures
# Let Ck denote a Convolution-BatchNorm-ReLU layer
# with k filters. CDk denotes a Convolution-BatchNormDropout-ReLU
# layer with a dropout rate of 50%. All convolutions are 4× 4 spatial
# filters applied with stride 2. Convolutions in the encoder, and in the discriminator, downsample
# by a factor of 2, whereas in the decoder they upsample by a
# factor of 2.


class EncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=2,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.encode(x)


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=4, stride=2,
                                     padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(), ]
        if use_dropout:
            layers.append(nn.Dropout())
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)
