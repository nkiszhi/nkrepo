import torch
import torch.nn as nn
import torch.nn.functional as F


class Malconv(nn.Module):
    def __init__(self, max_len=200000, win_size=500, vocab_size=256):
        super(Malconv, self).__init__()
        # Embedding layer: 将原始字节映射为 8 维嵌入向量
        self.embedding = nn.Embedding(vocab_size, 8)

        # 两层 1D 卷积层
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=win_size, stride=win_size, padding=0)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=win_size, stride=win_size, padding=0)

        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 确保输入为整数类型，适配embedding层
        x = x.long()

        # 嵌入层
        x = self.embedding(x)  # 输出形状: (batch_size, max_len, embedding_dim)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, embedding_dim, max_len) 适配 1D 卷积

        # 卷积 + 门控机制
        conv1_out = self.conv1(x)  # 主卷积输出
        conv2_out = self.conv2(x)  # 门控卷积输出
        gate = torch.sigmoid(conv2_out)  # 门控权重
        gated = conv1_out * gate  # 乘法操作完成门控机制

        # ReLU 激活
        activated = F.relu(gated)

        # 全局时间最大池化
        pooled = F.adaptive_max_pool1d(activated, 1).squeeze(-1)  # 输出形状: (batch_size, 128)

        # 全连接层
        fc1_out = F.relu(self.fc1(pooled))  # 输出形状: (batch_size, 64)
        output = torch.sigmoid(self.fc2(fc1_out))  # 输出形状: (batch_size, 1)

        return output
