import torch
import torch.nn as nn

class StandardConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding="same", dilation=1, groups=1):
        super(StandardConv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU() 
        ])

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv = MobileBlock(in_channel, out_channel, kernel_size, stride, dilation)

    def forward(self, x):
        return x + self.conv(x)

class MobileBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1):
        super(MobileBlock, self).__init__()
        pad = (kernel_size + (dilation - 1) * (kernel_size - 1) - stride) // 2
        self.conv_w = StandardConv(in_channel=in_channel, out_channel=in_channel, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channel, padding=pad)
        self.conv_d = StandardConv(in_channel=in_channel, out_channel=out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_w(x)
        x = self.conv_d(x)
        return x

class ResStandardBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1):
        super(ResStandardBlock, self).__init__()
        pad = (kernel_size + (dilation - 1) * (kernel_size - 1) - stride) // 2
        self.conv = StandardConv(in_channel, out_channel, kernel_size, stride, pad, dilation)

    def forward(self, x):
        return x + self.conv(x)