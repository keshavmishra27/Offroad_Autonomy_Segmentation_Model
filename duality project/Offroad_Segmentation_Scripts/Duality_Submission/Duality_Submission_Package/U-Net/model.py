import torch
import torch.nn as nn
from config import NUM_CLASSES

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):

        skip1 = self.down1(x)
        x = self.pool1(skip1)

        skip2 = self.down2(x)
        x = self.pool2(skip2)

        skip3 = self.down3(x)
        x = self.pool3(skip3)

        skip4 = self.down4(x)
        x = self.pool4(skip4)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = torch.cat((skip4, x), dim=1)
        x = self.decoder1(x)

        x = self.up2(x)
        x = torch.cat((skip3, x), dim=1)
        x = self.decoder2(x)

        x = self.up3(x)
        x = torch.cat((skip2, x), dim=1)
        x = self.decoder3(x)

        x = self.up4(x)
        x = torch.cat((skip1, x), dim=1)
        x = self.decoder4(x)

        x = self.out_conv(x)
        return x
