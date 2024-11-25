import torch
import torch.nn as nn
from SeqNet_main.models.regionnet.regionops import *

class RegionNet(nn.Module):
    def __init__(self, feat_num=16, input_size=(512, 512, 1), num_class=2):
        super(RegionNet, self).__init__()

        self.conv1_s3 = nn.Conv2d(in_channels=input_size[2], out_channels=feat_num, kernel_size=3, padding=1)

        # 512 * 512 * 16
        self.conv = nn.Sequential(*[
            nn.AvgPool2d(2),
            MobileBlock(in_channel=feat_num, out_channel=feat_num * 2, kernel_size=3),

            # 256 * 256 * 32
            nn.AvgPool2d(4),
            MobileBlock(in_channel=feat_num * 2, out_channel=feat_num * 4, kernel_size=3),

            # 64 * 64 * 64
            nn.AvgPool2d(2),
            MobileBlock(in_channel=feat_num * 4, out_channel=feat_num * 8, kernel_size=3),

            ResBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),

            # 32 * 32 * 128
            nn.AvgPool2d(2),
            MobileBlock(in_channel=feat_num * 8, out_channel=feat_num * 16, kernel_size=3),
        ])

        # 16 * 16 * 256
        self.conv_avg = nn.AvgPool2d(8)

        # 2 * 2 * 256

        self.fulc = nn.Linear(1024, num_class)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1_s3(x)
        x = self.conv(x)
        x = self.conv_avg(x)
        x = self.flatten(x)
        x = self.fulc(x)
        if not self.training:
            x = self.softmax(x)
        return x

class RegionNetConv(nn.Module):
    def __init__(self, feat_num=16, input_size=(512, 512, 1), num_class=2):
        super(RegionNetConv, self).__init__()

        self.conv1_s3 = nn.Conv2d(in_channels=input_size[2], out_channels=feat_num, kernel_size=3, padding=1)

        # 512 * 512 * 16
        self.conv = nn.Sequential(*[
            nn.AvgPool2d(2),
            StandardConv(in_channel=feat_num, out_channel=feat_num * 2, kernel_size=3, padding=1),

            # 256 * 256 * 32
            nn.AvgPool2d(4),
            StandardConv(in_channel=feat_num * 2, out_channel=feat_num * 4, kernel_size=3, padding=1),

            # 64 * 64 * 64
            nn.AvgPool2d(4),
            StandardConv(in_channel=feat_num * 4, out_channel=feat_num * 8, kernel_size=3, padding=1),

            ResStandardBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResStandardBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResStandardBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResStandardBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResStandardBlock(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),

            # 16 * 16 * 128
            nn.AvgPool2d(4),
            StandardConv(in_channel=feat_num * 8, out_channel=feat_num * 16, kernel_size=3, padding=1),
        ])

        # 4 * 4 * 256
        self.conv_avg = nn.AvgPool2d(2)

        # 2 * 2 * 256

        self.fulc = nn.Linear(1024, num_class)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1_s3(x)
        x = self.conv(x)
        x = self.conv_avg(x)
        x = self.flatten(x)
        x = self.fulc(x)
        if not self.training:
            x = torch.softmax(x, dim=1)
        return x