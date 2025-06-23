import torch
import torch.nn as nn
import torch.nn.functional as F
import models.m_2021_malconv2.checkpoint


from models.m_2021_malconv2.LowMemConv import LowMemConvBase
from models.m_2021_malconv2.MalConvML import MalConvML

class MalConvGCT(LowMemConvBase):
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None, low_mem=True):
        """
        初始化 MalConvGCT 模型。

        参数:
        out_size (int): 输出的类别数量。
        channels (int): 卷积层的通道数。
        window_size (int): 卷积窗口的大小。
        stride (int): 卷积的步长。
        layers (int): 卷积层的数量。
        embd_size (int): 嵌入层的维度。
        log_stride (int, 可选): 步长的对数表示。如果提供，将使用 2**log_stride 作为步长。
        low_mem (bool): 是否启用低内存模式。
        """
        super(MalConvGCT, self).__init__()
        self.low_mem = low_mem
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)

        # 动态调整步长
        if log_stride is not None:
            stride = 2 ** log_stride

        self.context_net = MalConvML(out_size=channels, channels=channels, window_size=window_size, stride=stride, layers=layers, embd_size=embd_size)
        self.convs = self._build_conv_layers(embd_size, channels, window_size, stride, layers)
        self.linear_atn = nn.ModuleList([nn.Linear(channels, channels) for _ in range(layers)])
        self.convs_share = nn.ModuleList([nn.Conv1d(channels, channels, 1, bias=True) for _ in range(layers)])
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)

    def _build_conv_layers(self, embd_size, channels, window_size, stride, layers):
        """
        构建卷积层模块列表。

        参数:
        embd_size (int): 嵌入层的维度。
        channels (int): 卷积层的通道数。
        window_size (int): 卷积窗口的大小。
        stride (int): 卷积的步长。
        layers (int): 卷积层的数量。

        返回:
        nn.ModuleList: 包含卷积层的模块列表。
        """
        convs = [nn.Conv1d(embd_size, channels * 2, window_size, stride=stride, bias=True)]
        for _ in range(layers - 1):
            convs.append(nn.Conv1d(channels, channels * 2, window_size, stride=1, bias=True))
        return nn.ModuleList(convs)

    def determinRF(self):
        """
        确定模型的感受野。

        返回:
        tuple: 包含感受野、步长和输出通道数的元组。
        """
        return self.context_net.determinRF()

    def processRange(self, x, gct=None, **kwargs):
        """
        处理输入数据的一个范围，应用卷积和全局上下文门控。

        参数:
        x (torch.Tensor): 输入的张量，形状为 (B, L)。
        gct (torch.Tensor, 可选): 全局上下文张量，形状为 (B, C)。
        **kwargs: 其他关键字参数。

        返回:
        torch.Tensor: 处理后的张量，形状为 (B, C, L)。
        """
        if gct is None:
            raise ValueError("No Global Context Given")

        x = self.embd(x)
        x = x.permute(0, 2, 1)

        for conv_glu, linear_cntx, conv_share in zip(self.convs, self.linear_atn, self.convs_share):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            B, C, _ = x.shape

            ctnx = torch.tanh(linear_cntx(gct))
            ctnx = ctnx.unsqueeze(2)
            x_tmp = x.view(1, B * C, -1)
            x_gates = F.conv1d(x_tmp, ctnx, groups=B).view(B, 1, -1)
            gates = torch.sigmoid(x_gates)
            x = x * gates

        return x

    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入的张量，形状为 (B, L)。

        返回:
        tuple: 包含输出、中间特征和卷积后特征的元组。
        """
        if self.low_mem:
            global_context = checkpoint.CheckpointFunction.apply(self.context_net.seq2fix, 1, x)
        else:
            global_context = self.context_net.seq2fix(x)

        post_conv = self.seq2fix(x, pr_args={'gct': global_context})
        penult = F.leaky_relu(self.fc_1(post_conv))
        output = self.fc_2(penult)

        return output, penult, post_conv