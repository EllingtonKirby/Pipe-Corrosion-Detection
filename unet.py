""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_steps=2, bilinear=False,with_pl=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_steps = n_steps

        self.inc = (DoubleConv(n_channels, 64))
        factor = 2 if bilinear else 1
        self.down1 = (Down(64, 128))
        self.up4 = (Up(128, 64, bilinear))
        self.down2 = (Down(128, 256))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.down3 = (Down(256, 512))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.with_pl = with_pl
        self.pseudo_label = nn.Sequential(
            nn.MaxPool2d(2), # 1x1
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features=1024//factor, out_features=1024//factor),
            nn.BatchNorm1d(num_features=1024//factor),
            nn.Linear(in_features=1024//factor, out_features=1),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = x2
        x3 = self.down2(x2)
        x = x3
        x4 = self.down3(x3)
        x = x4
        x5 = self.down4(x4)
        x = x5

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
    
        logits = self.outc(x)
        if self.with_pl:
            pseudo_label = self.pseudo_label(x5)
        else:
            pseudo_label = None

        latent_features = x5

        return logits, pseudo_label, latent_features

class Unet_Discriminator(nn.Module):
    def __init__(self, n_channels=1024, n_classes=1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features=n_channels, out_features=n_channels),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
            nn.Linear(in_features=n_channels, out_features=n_classes),
        )

    def forward(self, input):
        pred = self.classifier(input)
        return F.sigmoid(pred)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
