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

class OutEdge(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(OutEdge, self).__init__()
        self.conv1 = BasicConv2d(in_channel1, out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_channel2, out_channel, kernel_size=1)
        self.conv3 = BasicConv2d(out_channel, out_channel, kernel_size=1)

    def forward(self, foreg, backg):
        foreg = self.conv1(foreg)
        backg = self.conv2(backg)
        edge = foreg - backg
        edge = self.conv3(edge)

        return edge