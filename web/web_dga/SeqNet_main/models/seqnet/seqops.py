import torch.nn as nn

class StandardConv1d(nn.Module): # 卷积神经网络
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(StandardConv1d, self).__init__()

        # Sequential用于将多个网络层按照顺序串联起来，构建更复杂的网络模型
        self.conv = nn.Sequential(*[
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(),
        ])

    def forward(self, x):
        return self.conv(x)


class MobileBlock1d(nn.Module): # 深度可分离卷积 SDSC
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1):
        super(MobileBlock1d, self).__init__()
        pad = (kernel_size + (dilation - 1) * (kernel_size - 1) - stride) // 2
        # 宽度卷积，保持与输入相同的通道数 逐通道卷积groups=in_channel的时候表示做逐通道卷积
        self.conv_w = StandardConv1d(in_channel=in_channel, out_channel=in_channel, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channel, padding=pad)
        # 深度卷积，改变通道数 逐点卷积
        self.conv_d = StandardConv1d(in_channel=in_channel, out_channel=out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_w(x)
        x = self.conv_d(x)
        return x

class ResMobileBlock1d(nn.Module): # 残差深度可分离卷积神经网络
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1):
        super(ResMobileBlock1d, self).__init__()
        # 残差块，解决深度神经网络中梯度消失的问题，将输入和经过卷积处理的结果相加，形成一个跳跃连接
        self.conv = MobileBlock1d(in_channel, out_channel, kernel_size, stride, dilation)

    def forward(self, x):
        return x + self.conv(x)


class ResBlock1d(nn.Module): # 残差卷积神经网络
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1):
        super(ResBlock1d, self).__init__()
        pad = (kernel_size + (dilation - 1) * (kernel_size - 1) - stride) // 2
        self.conv = StandardConv1d(in_channel, out_channel, kernel_size, stride, pad, dilation)

    def forward(self, x):
        return x + self.conv(x)