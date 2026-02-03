import numpy as np
import torch
from pathlib import Path

import torch.nn as nn

class Ember(nn.Module):
    """与训练一致的模型结构"""
    def __init__(self, input_dim=2381, num_trees=10, tree_depth=3, output_dim=1):
        super().__init__()
        self.trees = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 2**tree_depth),
                nn.ReLU(),
                nn.Linear(2**tree_depth, output_dim))
            for _ in range(num_trees)
        ])
        self.global_weights = nn.Linear(num_trees, 1)
        
    def forward(self, x):
        outputs = [tree(x) for tree in self.trees]
        stacked = torch.stack(outputs, dim=-1)
        return self.global_weights(stacked).squeeze()
        
def extract_ember_features(file_path: str) -> np.ndarray:
    """示例特征提取（需替换为实际实现）"""
    try:
        # 实际应使用lief库提取特征
        return np.random.rand(2381)  # 生成随机示例数据
    except:
        return np.zeros(2381)

def run_prediction(model, file_path, device, scaler=None):
    """标准化预测接口"""
    try:
        features = extract_ember_features(file_path)
        if scaler:
            features = scaler.transform([features])
        inputs = torch.tensor(features, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            prob = torch.sigmoid(outputs).item()
        return None, prob, 1 if prob >= 0.5 else 0
    except Exception as e:
        print(f"[Ember错误] {str(e)}")
        return None, 0.0, -1