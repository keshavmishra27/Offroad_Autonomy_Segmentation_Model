import torch
import torch.nn as nn
from config import NUM_CLASSES

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES):
        super(SegNet, self).__init__()

        self.enc_conv1 = nn.Sequential(
            ConvBNReLU(in_channels, 64),
            ConvBNReLU(64, 64)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc_conv2 = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc_conv3 = nn.Sequential(
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc_conv4 = nn.Sequential(
            ConvBNReLU(256, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc_conv5 = nn.Sequential(
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv5 = nn.Sequential(
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512)
        )

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv4 = nn.Sequential(
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 256)
        )

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 128)
        )

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 64)
        )

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            ConvBNReLU(64, 64)
        )

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.enc_conv1(x)
        size1 = x.size()
        x, ind1 = self.pool1(x)

        x = self.enc_conv2(x)
        size2 = x.size()
        x, ind2 = self.pool2(x)

        x = self.enc_conv3(x)
        size3 = x.size()
        x, ind3 = self.pool3(x)

        x = self.enc_conv4(x)
        size4 = x.size()
        x, ind4 = self.pool4(x)

        x = self.enc_conv5(x)
        size5 = x.size()
        x, ind5 = self.pool5(x)

        x = self.unpool5(x, ind5, output_size=size5)
        x = self.dec_conv5(x)

        x = self.unpool4(x, ind4, output_size=size4)
        x = self.dec_conv4(x)

        x = self.unpool3(x, ind3, output_size=size3)
        x = self.dec_conv3(x)

        x = self.unpool2(x, ind2, output_size=size2)
        x = self.dec_conv2(x)

        x = self.unpool1(x, ind1, output_size=size1)
        x = self.dec_conv1(x)

        x = self.out_conv(x)
        return x
