import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    """
    基于 Transformer 的恶意软件分类模型
    """
    def __init__(self, seq_len=512, embed_dim=128, num_heads=8, ff_dim=256, num_layers=4, num_classes=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        # 嵌入层（字节值嵌入 + 位置嵌入）
        self.token_embedding = nn.Embedding(256, embed_dim)  # 假设输入为 0-255 的字节值
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

        # Transformer 编码器
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 分类头
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes)
        )

    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        batch_size, seq_len = x.size()

        # 生成位置索引
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # 嵌入层
        x = self.token_embedding(x) + self.position_embedding(positions)

        # 转换为 (seq_len, batch_size, embed_dim) 以适配 Transformer
        x = x.permute(1, 0, 2)

        # Transformer 编码器
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # 转换回 (batch_size, seq_len, embed_dim)
        x = x.permute(1, 0, 2)

        # 全局平均池化
        x = self.global_avg_pool(x.transpose(1, 2)).squeeze(-1)

        # 分类头
        output = self.fc(x)
        return output