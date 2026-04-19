import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.b5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        feat5 = self.b5(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=False)

        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.project(out)
        return out

class MobileDeepLab(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileDeepLab, self).__init__()

        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features

        self.low_level_features = backbone[:4]

        self.high_level_features = backbone[4:18]

        self.aspp = ASPP(in_channels=320, out_channels=256)

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv = nn.Sequential(
            DepthwiseSeparableConv(304, 256),
            DepthwiseSeparableConv(256, 256)
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]

        low_level = self.low_level_features(x)
        high_level = self.high_level_features(low_level)

        aspp_out = self.aspp(high_level)

        aspp_out = F.interpolate(aspp_out, scale_factor=4, mode='bilinear', align_corners=False)

        low_level = self.low_level_conv(low_level)

        if aspp_out.shape[-2:] != low_level.shape[-2:]:
            aspp_out = F.interpolate(aspp_out, size=low_level.shape[-2:], mode='bilinear', align_corners=False)

        concat = torch.cat([aspp_out, low_level], dim=1)

        dec_out = self.decoder_conv(concat)

        out = self.classifier(dec_out)

        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out
