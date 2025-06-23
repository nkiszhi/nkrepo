# m_rcnn.py
import os
import logging
import torch
import pefile
from pathlib import Path
from torch import nn
from typing import List, Tuple, Optional
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

class RCNN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        out_channels: int = 64,
        window_size: int = 3,
        module: nn.Module = nn.LSTM,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        residual: bool = True,
        dropout: float = 0.5,
    ):
        super(RCNN, self).__init__()
        assert module.__name__ in {"RNN", "GRU", "LSTM"}, "`module` 必须是 PyTorch 循环层"
        self.residual = residual
        self.embed = nn.Embedding(257, embed_dim)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=window_size,
            padding=1
        )
        self.rnn = module(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=False
        )
        self.dropout = nn.Dropout(dropout)
        rnn_out_size = (int(bidirectional) + 1) * hidden_size
        self.fc = nn.Linear(out_channels + rnn_out_size, 1) if residual else nn.Linear(rnn_out_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入验证
        if x.dim() != 2 or x.size(1) < 3:
            raise ValueError(f"无效输入形状: {x.shape}, 至少需要3个特征")
            
        embedding = self.dropout(self.embed(x))
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        values, _ = conv_out.max(dim=-1) if self.residual else (None, None)
        conv_out = conv_out.permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv_out)
        fc_in = rnn_out[-1]
        if self.residual:
            fc_in = torch.cat((fc_in, values), dim=-1)
        return self.fc(fc_in).squeeze(1)

def scan_load_samples(samples_dir: str) -> List[Tuple[str, int]]:
    """训练模式加载样本"""
    samples = []
    path = Path(samples_dir)
    for file_path in path.glob("*"):
        # 假设文件名包含标签信息，例如：malware_1.exe, benign_0.exe
        label = 1 if "malware" in file_path.name else 0
        samples.append((str(file_path), label))
    return samples

def scan_load_prediction_samples(input_path: str) -> List[str]:
    """预测模式加载样本（支持目录/文件路径）"""
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    return [str(p) for p in path.glob("*") if p.is_file()]

def extract_features_rcnn(samples: List) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    统一的特征提取函数
    Args:
        samples: 训练时为 [(file_path, label)], 预测时为 [file_path]
    """
    MIN_LENGTH = 3  # 与卷积核大小匹配
    all_features = []
    all_labels = []

    for item in samples:
        file_path = item[0] if isinstance(item, tuple) else item
        try:
            pe = pefile.PE(file_path, fast_load=True)
            
            # 提取基础特征
            entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint if hasattr(pe, 'OPTIONAL_HEADER') else 0
            num_sections = len(pe.sections)
            section_sizes = [s.SizeOfRawData % 256 + 1 for s in pe.sections]
            section_entropies = [int(s.get_entropy() * 100) % 256 + 1 for s in pe.sections]
            
            # 组合特征序列
            features = [num_sections, entry_point] + section_sizes + section_entropies
            features = [v % 257 for v in features]  # 确保在0-256范围内
            
            # 填充/截断处理
            if len(features) < MIN_LENGTH:
                features += [0] * (MIN_LENGTH - len(features))
            else:
                features = features[:512]  # 限制最大长度
                
            all_features.append(torch.tensor(features, dtype=torch.long))
            
            # 处理标签
            if isinstance(item, tuple):
                all_labels.append(torch.tensor([item[1]], dtype=torch.float32))

        except Exception as e:
            logger.error(f"特征提取失败: {file_path} - {str(e)}")
            continue

    # 处理空特征情况
    if not all_features:
        return torch.zeros((0, MIN_LENGTH), dtype=torch.long), None
        
    features_tensor = torch.stack(all_features)
    labels_tensor = torch.cat(all_labels) if all_labels else None
    return features_tensor, labels_tensor

def run_prediction(model: nn.Module, file_path: str, device: torch.device) -> Tuple[None, float, int]:
    """
    标准预测接口（与集成系统兼容）
    返回: (None, 概率值, 0/1标签)
    """
    try:
        # 加载样本并提取特征
        samples = scan_load_prediction_samples(file_path)
        features, _ = extract_features_rcnn(samples)
        
        if features.size(0) == 0:
            logger.warning(f"无有效特征: {file_path}")
            return None, 0.0, 0
            
        # 维度调整
        features = features.to(device)
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        # 执行预测
        with torch.no_grad():
            output = model(features)
            prob = torch.sigmoid(output).item()
            
        return None, prob, int(prob >= 0.5)
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}", exc_info=True)
        return None, 0.0, -1

# 训练函数（保持原始训练逻辑）
def running_training():
    SAMPLES_DIR = r"E:\Experimental data\dr_data"
    MODEL_PATH = "rcnn_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    samples = scan_load_samples(SAMPLES_DIR)
    features, labels = extract_features_rcnn(samples)
    
    # 数据预处理
    features = features.clamp(0, 256)
    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 模型初始化
    model = RCNN(
        embed_dim=128,
        out_channels=64,
        window_size=3,
        module=nn.LSTM,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        residual=True
    ).to(device)

    # 训练循环
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至 {MODEL_PATH}")

if __name__ == "__main__":
    # 训练模式
    # running_training()
    
    # 预测模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN().to(device)
    model.load_state_dict(torch.load("rcnn_model.pth", map_location=device))
    model.eval()
    
    test_file = "/path/to/test_file.exe"
    _, prob, label = run_prediction(model, test_file, device)
    print(f"预测结果: 概率={prob:.4f}, 标签={'恶意' if label else '良性'}")


