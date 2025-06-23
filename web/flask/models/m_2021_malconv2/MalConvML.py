import torch
import torch.nn as nn
import torch.nn.functional as F
from models.m_2021_malconv2.LowMemConv import LowMemConvBase

class MalConvML(LowMemConvBase):
    """
    多层层叠 MalConv 模型，继承自 LowMemConvBase 实现低内存分块处理。

    参数:
    - out_size: 输出类别数（默认2）
    - channels: 卷积通道数（默认128）
    - window_size: 卷积窗口大小（默认512）
    - stride: 卷积步长（默认512）
    - layers: 卷积层数（默认1）
    - embd_size: 嵌入维度（默认8）
    - log_stride: 步长的对数（替代 stride 参数）
    """

    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None):
        super(MalConvML, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)

        # 动态步长计算
        if log_stride is not None:
            stride = 2 ** log_stride

        # 构建卷积层
        self.convs = self._build_convs(embd_size, channels, window_size, stride, layers)
        self.convs_1 = nn.ModuleList([
            nn.Conv1d(channels, channels, 1, bias=True)
            for _ in range(layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, out_size)
        )

    def _build_convs(self, embd_size, channels, window_size, stride, layers):
        """
        构建多层卷积层

        参数:
        - embd_size: 嵌入维度
        - channels: 通道数
        - window_size: 窗口大小
        - stride: 初始步长
        - layers: 层数

        返回:
        - ModuleList: 卷积层列表
        """
        convs = [
            nn.Conv1d(embd_size, channels * 2, window_size, stride=stride, bias=True)
        ]
        for _ in range(layers - 1):
            convs.append(
                nn.Conv1d(channels, channels * 2, window_size, stride=1, bias=True)
            )
        return nn.ModuleList(convs)

    def processRange(self, x, **kwargs):
        r"""
        特征提取核心逻辑

        参数:
        - x: 输入张量 (B, L)

        返回:
        - features: 特征张量 (B, C, L)
        """
        x = self.embd(x)  # (B, L) -> (B, L, E)
        x = x.permute(0, 2, 1).contiguous()  # (B, E, L)

        for conv, conv_share in zip(self.convs, self.convs_1):
            x = F.glu(conv(x), dim=1)  # (B, C*2, L) -> (B, C, L)
            x = F.leaky_relu(conv_share(x))  # 信息共享

        return x

    def forward(self, x):
        """
        前向传播

        参数:
        - x: 输入张量 (B, L)

        返回:
        - logits: 分类结果 (B, out_size)
        - penult: 中间特征 (B, C)
        - post_conv: 卷积后特征 (B, C, L)
        """
        # 分块处理
        post_conv = self.seq2fix(x)  # (B, C)

        # 分类头
        penult = self.classifier[0](post_conv)  # (B, C)
        logits = self.classifier[2](F.relu(penult))  # (B, out_size)

        return logits, penult, post_conv

    def _get_device(self):
        """
        获取模型所在设备
        """
        return next(self.parameters()).device if self.parameters() else torch.device("cpu")