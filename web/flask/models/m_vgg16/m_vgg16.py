# models/m_vgg16/m_vgg16.py
import os
import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch import nn
from typing import Tuple, Optional
import torchvision.models as models
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class VGG16Malware(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(VGG16Malware, self).__init__()
        self.features = models.vgg16(weights=None).features  # 关键修改
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def _pe_to_image(file_path: str, image_size: tuple = (224, 224)) -> Optional[Image.Image]:
    """将PE文件转换为灰度图像"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        byte_array = np.frombuffer(data, dtype=np.uint8)
        side_length = int(np.ceil(np.sqrt(len(byte_array))))
        padded_array = np.pad(byte_array, (0, side_length**2 - len(byte_array)), mode='constant')
        return Image.fromarray(padded_array.reshape(side_length, side_length), mode='L').resize(image_size)
    except Exception as e:
        logger.error(f"PE转换图像失败: {file_path} - {str(e)}")
        return None

def extract_features_vgg16(file_path: str) -> Optional[torch.Tensor]:
    """统一特征提取接口"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = _pe_to_image(file_path)
        if image is None:
            return None
        return transform(image).unsqueeze(0)  # 添加batch维度
    except Exception as e:
        logger.error(f"特征提取失败: {file_path} - {str(e)}")
        return None

def run_prediction(model: nn.Module, file_path: str, device: torch.device) -> Tuple[None, float, int]:
    """
    标准预测接口
    返回: (None, 概率值, 0/1标签)
    """
    try:
        # 特征提取
        features = extract_features_vgg16(file_path)
        if features is None:
            return None, 0.0, 0
        
        # 数据预处理
        features = features.to(device)
        
        # 执行预测
        with torch.no_grad():
            outputs = model(features)
            prob = torch.softmax(outputs, dim=1)[0, 1].item()
            
        return None, prob, int(prob >= 0.5)
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}", exc_info=True)
        return None, 0.0, -1

# 训练函数（保持原始逻辑）
def running_training():
    """训练入口函数"""
    from torch.utils.data import Dataset, DataLoader
    
    class PEImageDataset(Dataset):
        def __init__(self, samples_dir: str):
            self.samples = []
            for root, _, files in os.walk(samples_dir):
                for file in files:
                    label = 1 if "malware" in file.lower() else 0
                    self.samples.append((os.path.join(root, file), label))
                    
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            file_path, label = self.samples[idx]
            features = extract_features_vgg16(file_path)
            return features.squeeze(0), torch.tensor(label, dtype=torch.long)  # 移除batch维度

    dataset = PEImageDataset(r"E:\Experimental data\dr_data")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VGG16Malware().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练循环...
    # (保持原始训练逻辑)

if __name__ == "__main__":
    # 示例用法
    device = torch.device("cpu")
    model = VGG16Malware()
    model.load_state_dict(torch.load("trained_vgg16_model.pth", map_location=device))
    model.eval()
    
    test_file = "/path/to/test.exe"
    _, prob, label = run_prediction(model, test_file, device)
    print(f"预测结果: 概率={prob:.4f}, 标签={'恶意' if label else '良性'}")