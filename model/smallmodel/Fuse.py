import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Fuse(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super(Fuse, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_upsample4_3 = BasicConv2d(c4, c3, 3, padding=1)
        self.conv_upsample4_2 = BasicConv2d(c4, c2, 3, padding=1)
        self.conv_upsample4_1 = BasicConv2d(c4, c1, 3, padding=1)
        self.conv_upsample3_2 = BasicConv2d(c3, c2, 3, padding=1)
        self.conv_upsample3_1 = BasicConv2d(c3, c1, 3, padding=1)
        self.conv_upsample2_1 = BasicConv2d(c2, c1, 3, padding=1)

        self.conv4_up = BasicConv2d(c4, c3, 3, padding=1)
        self.conv3_4_up = BasicConv2d(c3, c2, 3, padding=1)
        self.conv2_3_up = BasicConv2d(c2, c1, 3, padding=1)

        self.conv_concat3_4 = BasicConv2d(c3 + c3, c3, 3, padding=1)
        self.conv_concat2_3 = BasicConv2d(c2 + c2, c2, 3, padding=1)
        self.conv_concat1_2 = BasicConv2d(c1 + c1, 128, 3, padding=1)

    def forward(self, x1, x2, x3, x4):
        x4_1 = x4
        x3_1 = self.conv_upsample4_3(self.upsample2(x4)) * x3
        x2_1 = self.conv_upsample4_2(self.upsample4(x4)) \
               * self.conv_upsample3_2(self.upsample2(x3)) * x2
        x1_1 = self.conv_upsample4_1(self.upsample8(x4)) * self.conv_upsample3_1(self.upsample4(x3)) \
               * self.conv_upsample2_1(self.upsample2(x2)) * x1

        x3_4 = torch.cat((x3_1, self.conv4_up(self.upsample2(x4_1))), 1)
        x3_4 = self.conv_concat3_4(x3_4)

        x2_3 = torch.cat((x2_1, self.conv3_4_up(self.upsample2(x3_4))), 1)
        x2_3 = self.conv_concat2_3(x2_3)

        x1_2 = torch.cat((x1_1, self.conv2_3_up(self.upsample2(x2_3))), 1)
        x = self.conv_concat1_2(x1_2)

        return x