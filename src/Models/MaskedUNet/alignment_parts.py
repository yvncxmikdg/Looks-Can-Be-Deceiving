import torch

from torch import nn
import torch.nn.functional as F

class AlignmentDown(nn.Module):
    def __init__(self, in_channels, dilation=1, kernel_size=5, padding_mode="reflect"):
        super().__init__()
        padding = (kernel_size//2)*dilation
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False, padding_mode=padding_mode),
            nn.GroupNorm(in_channels, in_channels),
            nn.Conv2d(in_channels, 2, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False, padding_mode=padding_mode),
        )

    def forward(self, x):
        return self.double_conv(x)

class AlignmentUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pylint: disable=not-callable
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode="reflect")
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x
