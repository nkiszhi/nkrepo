import torch
import torch.nn as nn
from SeqNet_main.models.seqnet.seqops import *

class SequenceNet(nn.Module):
    def __init__(self, in_length=2 ** 18, feat_num=16, num_class=2):
        super(SequenceNet, self).__init__()
        # 1 * 2 ** 18
        # para 48
        self.conv1_s3 = nn.Conv1d(in_channels=1, out_channels=feat_num, kernel_size=3, padding=1)

        # 16 * 2 ** 18
        self.conv = nn.Sequential(*[
            nn.AvgPool1d(4),
            # nn.MaxPool1d(4),
            MobileBlock1d(in_channel=feat_num, out_channel=feat_num * 2, kernel_size=3),

            # 32 * 2 ** 16
            nn.AvgPool1d(8),
            # nn.MaxPool1d(8),
            MobileBlock1d(in_channel=feat_num * 2, out_channel=feat_num * 4, kernel_size=3),

            # 64 * 2 ** 13
            nn.AvgPool1d(8),
            # nn.MaxPool1d(8),
            MobileBlock1d(in_channel=feat_num * 4, out_channel=feat_num * 8, kernel_size=3),

            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),

            # 128 * 2 ** 10
            nn.AvgPool1d(8),
            # nn.MaxPool1d(8),
            MobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 16, kernel_size=3),
        ])

        # 256 * 2 ** 7
        self.conv_avg = nn.AvgPool1d(32)

        # 256 * 2 ** 2

        # para 2048
        self.fulc = nn.Linear(in_length * feat_num * 16 // (2 ** 16), num_class)
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

# 不使用深度可分离卷积MobileBlock1d 使用普通卷积StandardConv1d
class SequenceNetConv(nn.Module):
    def __init__(self, in_length=2 ** 18, feat_num=16, num_class=2):
        super(SequenceNetConv, self).__init__()
        # 1 * 2 ** 18
        # para 48
        self.conv1_s3 = nn.Conv1d(in_channels=1, out_channels=feat_num, kernel_size=3, padding=1)

        # 16 * 2 ** 18
        self.conv = nn.Sequential(*[
            nn.AvgPool1d(4),
            StandardConv1d(in_channel=feat_num, out_channel=feat_num * 2, kernel_size=3, padding=1),

            # 32 * 2 ** 16
            nn.AvgPool1d(8),
            StandardConv1d(in_channel=feat_num * 2, out_channel=feat_num * 4, kernel_size=3, padding=1),

            # 64 * 2 ** 13
            nn.AvgPool1d(8),
            StandardConv1d(in_channel=feat_num * 4, out_channel=feat_num * 8, kernel_size=3, padding=1),

            ResBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),

            # 128 * 2 ** 10
            nn.AvgPool1d(8),
            # nn.MaxPool1d(8)
            StandardConv1d(in_channel=feat_num * 8, out_channel=feat_num * 16, kernel_size=3, padding=1),
        ])

        # 256 * 2 ** 7
        self.conv_avg = nn.AvgPool1d(32)

        # 256 * 2 ** 2

        # para 2048
        self.fulc = nn.Linear(in_length * feat_num * 16 // (2 ** 16), num_class)
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

# 加了一层深度可分离卷积
class SequenceNetDeeper(nn.Module):
    def __init__(self, in_length=2 ** 18, feat_num=16, num_class=2):
        super(SequenceNetDeeper, self).__init__()
        # 1 * 2 ** 18
        # para 48
        self.conv1_s3 = nn.Conv1d(in_channels=1, out_channels=feat_num, kernel_size=3, padding=1)

        # 16 * 2 ** 18
        self.conv = nn.Sequential(*[
            nn.AvgPool1d(4),
            MobileBlock1d(in_channel=feat_num, out_channel=feat_num * 2, kernel_size=3),

            # 32 * 2 ** 16
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 2, out_channel=feat_num * 4, kernel_size=3),

            # 64 * 2 ** 13
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 4, out_channel=feat_num * 8, kernel_size=3),

            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),

            # 128 * 2 ** 10
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 16, kernel_size=3),

            # 256 * 2 ** 7
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 16, out_channel=feat_num * 32, kernel_size=3),
        ])

        # 512 * 2 ** 4
        self.conv_avg = nn.AvgPool1d(8)

        # 512 * 2 ** 1
        self.fulc = nn.Linear(in_length * feat_num * 16 // (2 ** 16), num_class)
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

# 少了一层深度可分离卷积
class SequenceNetShallow(nn.Module):
    def __init__(self, in_length=2 ** 18, feat_num=16, num_class=2):
        super(SequenceNetShallow, self).__init__()
        # 1 * 2 ** 18
        # para 48
        self.conv1_s3 = nn.Conv1d(in_channels=1, out_channels=feat_num, kernel_size=3, padding=1)

        # 16 * 2 ** 18
        self.conv = nn.Sequential(*[
            nn.AvgPool1d(4),
            MobileBlock1d(in_channel=feat_num, out_channel=feat_num * 2, kernel_size=3),

            # 32 * 2 ** 16
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 2, out_channel=feat_num * 4, kernel_size=3),

            # 64 * 2 ** 13
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 4, out_channel=feat_num * 8, kernel_size=3),

            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),

            # 128 * 2 ** 10
        ])

        # 128 * 2 ** 10
        self.conv_avg = nn.AvgPool1d(128)

        # 128 * 2 ** 3
        self.fulc = nn.Linear(in_length * feat_num * 16 // (2 ** 16), num_class)
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

# 没有残差块
class SequenceNetFool(nn.Module):
    def __init__(self, in_length=2 ** 18, feat_num=16, num_class=2):
        super(SequenceNetFool, self).__init__()
        # 1 * 2 ** 18
        # para 48
        self.conv1_s3 = nn.Conv1d(in_channels=1, out_channels=feat_num, kernel_size=3, padding=1)

        # 16 * 2 ** 18
        self.conv = nn.Sequential(*[
            nn.AvgPool1d(4),
            MobileBlock1d(in_channel=feat_num, out_channel=feat_num * 2, kernel_size=3),

            # 32 * 2 ** 16
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 2, out_channel=feat_num * 4, kernel_size=3),

            # 64 * 2 ** 13
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 4, out_channel=feat_num * 8, kernel_size=3),

            # 128 * 2 ** 10
            nn.AvgPool1d(8),
            MobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 16, kernel_size=3),
        ])

        # 256 * 2 ** 7
        self.conv_avg = nn.AvgPool1d(32)

        # 256 * 2 ** 2

        # para 2048
        self.fulc = nn.Linear(in_length * feat_num * 16 // (2 ** 16), num_class)
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


class SequenceNetMax(nn.Module):
    def __init__(self, in_length=2 ** 18, feat_num=16, num_class=2):
        super(SequenceNetMax, self).__init__()
        # 1 * 2 ** 18
        # para 48
        self.conv1_s3 = nn.Conv1d(in_channels=1, out_channels=feat_num, kernel_size=3, padding=1)

        # 16 * 2 ** 18
        self.conv = nn.Sequential(*[
            nn.MaxPool1d(4),
            # nn.MaxPool1d(4),
            MobileBlock1d(in_channel=feat_num, out_channel=feat_num * 2, kernel_size=3),

            # 32 * 2 ** 16
            nn.MaxPool1d(8),
            # nn.MaxPool1d(8),
            MobileBlock1d(in_channel=feat_num * 2, out_channel=feat_num * 4, kernel_size=3),

            # 64 * 2 ** 13
            nn.MaxPool1d(8),
            # nn.MaxPool1d(8),
            MobileBlock1d(in_channel=feat_num * 4, out_channel=feat_num * 8, kernel_size=3),

            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),
            ResMobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 8, kernel_size=3),

            # 128 * 2 ** 10
            nn.MaxPool1d(8),
            # nn.MaxPool1d(8),
            MobileBlock1d(in_channel=feat_num * 8, out_channel=feat_num * 16, kernel_size=3),
        ])

        # 256 * 2 ** 7
        self.conv_avg = nn.MaxPool1d(32)

        # 256 * 2 ** 2

        # para 2048
        self.fulc = nn.Linear(in_length * feat_num * 16 // (2 ** 16), num_class)
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