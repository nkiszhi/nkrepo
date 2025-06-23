import torch
import torch.nn as nn
from pathlib import Path

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len=512, embed_dim=128, num_heads=8, ff_dim=256, num_layers=4, num_classes=2, dropout=0.1):
        super().__init__()
        # 嵌入层（字节值嵌入 + 位置嵌入）
        self.token_embedding = nn.Embedding(256, embed_dim)
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
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        x = self.global_avg_pool(x.transpose(1, 2)).squeeze(-1)
        return self.fc(x)

# 修改 m_2017_transformer.py 中的 process_data 函数
def process_data(file_path: str) -> torch.Tensor:
    try:
        with open(file_path, "rb") as f:
            bytez = f.read()
        byte_values = list(bytez[:512])  # 与训练时的 SEQ_LEN=512 一致
        if len(byte_values) < 512:
            byte_values += [0] * (512 - len(byte_values))
        return torch.tensor(byte_values, dtype=torch.long).unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"数据处理失败: {str(e)}")

def run_prediction(model, file_path, device, scaler=None):
    """标准化预测接口"""
    try:
        # 确保模型在 CPU 上
        model = model.to('cpu')  # 新增代码
        
        inputs = process_data(file_path).to('cpu')  # 强制使用 CPU
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            prob = probs[0][1].item()
        return None, prob, 1 if prob >= 0.5 else 0
    except Exception as e:
        print(f"[Transformer错误] {str(e)}")
        return None, 0.0, -1