import torch
import torch.nn as nn
import torch.nn.functional as F


class PipeIdentifier(nn.Module):
  def __init__(self, num_classes) -> None:
    super(PipeIdentifier, self).__init__()
    self.encoder = nn.Sequential(
        DoubleConv(in_channels=1, out_channels=64), # 36x36
        Down(in_channels=64, out_channels=128), # 18x18
        Down(in_channels=128, out_channels=256), # 9x9
        Down(in_channels=256, out_channels=512), # 4x4
        Down(in_channels=512, out_channels=512), # 2x2
        nn.MaxPool2d(2), # 1x1
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(),
        nn.Linear(in_features=512, out_features=512),
        nn.BatchNorm1d(num_features=512),
        nn.Linear(in_features=512, out_features=num_classes),
    )

  def forward(self, input):
      x = self.encoder(input)
      logits = self.classifier(x)
      return logits

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
