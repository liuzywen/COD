import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from .basemodel.segformer import mit_b5
from .smallmodel.cross_transformer import Transformer
from .smallmodel.OutEdges import OutEdge
from .smallmodel.Fuse import Fuse

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

class mit_trans(nn.Module):
    def __init__(self, config, image_size, opt):
        super(mit_trans, self).__init__()
        self.encoder1 = mit_b5()
        self.encoder2 = mit_b5()
        self.out_edge1 = OutEdge(64, 64, 64)
        self.out_edge2 = OutEdge(128, 128, 128)
        self.out_edge3 = OutEdge(320, 320, 320)
        self.out_edge4 = OutEdge(512, 512, 512)
        self.fuse_edge = Fuse(64, 128, 320, 512)
        self.fuse_foreg = Fuse(64, 128, 320, 512)
        self.cross = Transformer(config, image_size, vis=False)
        self.conv0 = BasicConv2d(768, 128, kernel_size=1)
        self.conv1 = BasicConv2d(768, 128, kernel_size=1)
        self.conv2 = BasicConv2d(512, 128, kernel_size=1)
        self.conv3 = BasicConv2d(512, 128, kernel_size=1)
        self.conv4 = BasicConv2d(512, 128, kernel_size=1)
        self.conv5 = BasicConv2d(320, 128, kernel_size=1)
        self.conv6 = BasicConv2d(128, 64, kernel_size=1)
        self.conv7 = BasicConv2d(64, 32, kernel_size=1)
        self.conv8 = BasicConv2d(128, 64, kernel_size=1)
        self.conv9 = BasicConv2d(128, 64, kernel_size=1)
        self.pred0 = nn.Conv2d(128, 1, kernel_size=1)
        self.pred1 = nn.Conv2d(128, 1, kernel_size=1)
        self.pred2 = nn.Conv2d(128, 1, kernel_size=1)
        self.pred3 = nn.Conv2d(128, 1, kernel_size=1)
        self.pred4 = nn.Conv2d(128, 1, kernel_size=1)
        self.pred5 = nn.Conv2d(128, 1, kernel_size=1)
        self.pred6 = nn.Conv2d(64, 1, kernel_size=1)
        self.pred7 = nn.Conv2d(32, 1, kernel_size=1)
        self.pred8 = nn.Conv2d(64, 1, kernel_size=1)
        self.pred9 = nn.Conv2d(64, 1, kernel_size=1)
    def load_checkpoint(self, path):
        self.cross.load_from(weights=np.load(path))

    def forward(self, input):
        H, W = input.size()[2], input.size()[3]
        foregs = self.encoder1(input)
        backgs = self.encoder2(input)
        foreg1, foreg2, foreg3, foreg4 = foregs[0], foregs[1], foregs[2], foregs[3]
        backg1, backg2, backg3, backg4 = backgs[0], backgs[1], backgs[2], backgs[3]
        edge1 = self.out_edge1(foreg1, backg1)
        edge2 = self.out_edge2(foreg2, backg2)
        edge3 = self.out_edge3(foreg3, backg3)
        edge4 = self.out_edge4(foreg4, backg4)
        fcoee = self.fuse_edge(edge1, edge2, edge3, edge4)
        fcod = self.fuse_foreg(foreg1, foreg2, foreg3, foreg4)
        cod, coee = self.cross(fcod, fcoee)
        B, n_patch, hidden = cod.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        cod = cod.permute(0, 2, 1)
        coee = coee.permute(0, 2, 1)
        cod = cod.contiguous().view(B, hidden, h, w)
        coee = coee.contiguous().view(B, hidden, h, w)
        coee = self.pred0(self.conv0(coee))
        cod = self.pred1(self.conv1(cod))
        s_foreg = self.pred2(self.conv2(foreg4))
        s_backg = self.pred3(self.conv3(backg4))
        edge4 = self.pred4(self.conv4(edge4))
        edge3 = self.pred5(self.conv5(edge3))
        edge2 = self.pred6(self.conv6(edge2))
        edge1 = self.pred7(self.conv7(edge1))
        s_coee = self.pred8(self.conv8(fcoee))
        s_cod = self.pred9(self.conv9(fcod))
        coee = F.interpolate(coee, size=(H, W), mode='bilinear', align_corners=False)
        cod = F.interpolate(cod, size=(H, W), mode='bilinear', align_corners=False)
        s_foreg = F.interpolate(s_foreg, size=(H, W), mode='bilinear', align_corners=False)
        s_backg = F.interpolate(s_backg, size=(H, W), mode='bilinear', align_corners=False)
        edge4 = F.interpolate(edge4, size=(H, W), mode='bilinear', align_corners=False)
        edge3 = F.interpolate(edge3, size=(H, W), mode='bilinear', align_corners=False)
        edge2 = F.interpolate(edge2, size=(H, W), mode='bilinear', align_corners=False)
        edge1 = F.interpolate(edge1, size=(H, W), mode='bilinear', align_corners=False)
        s_coee = F.interpolate(s_coee, size=(H, W), mode='bilinear', align_corners=False)
        s_cod = F.interpolate(s_cod, size=(H, W), mode='bilinear', align_corners=False)
        return cod, coee, s_foreg, s_backg, edge4, edge3, edge2, edge1, s_coee, s_cod



