import torch
import torch.nn as nn
import numpy as np

def bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    Returns a legitimately calculated bilinear filter tensor for initializing ConvTranspose2d weight.
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

class VGG16Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(VGG16Encoder, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)
        pool3 = x

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)
        pool4 = x

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)
        pool5 = x

        return pool3, pool4, pool5

class FCN32s(nn.Module):
    def __init__(self, num_classes=10):
        super(FCN32s, self).__init__()
        self.encoder = VGG16Encoder()

        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)

        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, padding=16, bias=False)
        self.upscore.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 64))

    def forward(self, x):
        _, _, pool5 = self.encoder(x)
        x = self.relu6(self.fc6(pool5))
        x = self.relu7(self.fc7(x))
        x = self.score_fr(x)
        x = self.upscore(x)
        return x

class FCN16s(nn.Module):
    def __init__(self, num_classes=10):
        super(FCN16s, self).__init__()
        self.encoder = VGG16Encoder()

        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)

        self.score_fr = nn.Conv2d(4096, num_classes, 1)

        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)
        self.upscore2.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 4))

        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=16, padding=8, bias=False)
        self.upscore16.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 32))

    def forward(self, x):
        _, pool4, pool5 = self.encoder(x)

        f = self.relu6(self.fc6(pool5))
        f = self.relu7(self.fc7(f))
        score_fr = self.score_fr(f)

        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(pool4)

        fuse_pool4 = upscore2 + score_pool4

        x = self.upscore16(fuse_pool4)
        return x

class FCN8s(nn.Module):
    def __init__(self, num_classes=10):
        super(FCN8s, self).__init__()
        self.encoder = VGG16Encoder()

        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)

        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)
        self.upscore2.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 4))

        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)
        self.upscore_pool4.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 4))

        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4, bias=False)
        self.upscore8.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 16))

    def forward(self, x):
        pool3, pool4, pool5 = self.encoder(x)

        f = self.relu6(self.fc6(pool5))
        f = self.relu7(self.fc7(f))
        score_fr = self.score_fr(f)

        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(pool4)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(pool3)
        fuse_pool3 = upscore_pool4 + score_pool3

        x = self.upscore8(fuse_pool3)
        return x
