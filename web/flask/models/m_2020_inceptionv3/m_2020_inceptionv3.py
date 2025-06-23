# m_2020_inceptionv3.py
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from torchvision import transforms

logger = logging.getLogger(__name__)

# ================= 模型定义 =================
class InceptionV3Model(nn.Module):
    def __init__(self, num_classes=25):
        super().__init__()
        # 明确设置aux_logits=False，并保留子模块结构
        self.inception_v3 = models.inception_v3(weights=None, aux_logits=False,init_weights=True)
        num_ftrs = self.inception_v3.fc.in_features
        self.inception_v3.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.inception_v3(x)

# ================= 预处理工具 =================
class BinaryConverter:
    @staticmethod
    def convert(file_path: str, img_size=(299, 299)):
        try:
            with open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            
            target_length = img_size[0] * img_size[1] * 3
            if len(data) > target_length:
                data = data[:target_length]
            else:
                data = np.pad(data, (0, target_length - len(data)), mode='constant')

            img = data.reshape(img_size[0], img_size[1], 3).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img
        except Exception as e:
            logger.error(f"文件转换失败: {str(e)}")
            return None

# ================= 预测函数 =================
def run_prediction(model: nn.Module, file_path: str, device: torch.device):
    try:
        img_array = BinaryConverter.convert(file_path)
        if img_array is None:
            return False, 0.0, -1

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img_array).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        malicious_prob = probs[0][1].item()
        label = 1 if malicious_prob >= 0.5 else 0
        
        return True, malicious_prob, label

    except Exception as e:
        logger.error(f"预测异常: {str(e)}")
        return False, 0.0, -1

# ================= 模型加载修复 =================
if __name__ == "__main__":
    model = InceptionV3Model(num_classes=25)
    weight_path = Path(__file__).parent / "saved/inceptionv3_malware_model.pth"
    
    if weight_path.exists():
        # 加载原始状态字典（包含或不包含模块前缀）
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        
        # 修复键名：为所有键添加"inception_v3."前缀（适用于训练时未使用子模块的情况）
        new_state_dict = {}
        for old_key, value in state_dict.items():
            # 跳过辅助分类器参数（如果存在）
            if "AuxLogits" in old_key:
                continue
            
            # 添加子模块前缀
            new_key = f"inception_v3.{old_key}"
            new_state_dict[new_key] = value

        # 加载修复后的状态字典
        try:
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("✅ 模型参数加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}", exc_info=True)
            exit()
    else:
        logger.error("❌ 权重文件不存在")
        exit()

    # 测试预测
    test_file = "/home/user/MCDM/flask/models/m_2020_inceptionv3/data/2d0da2a2d7f379ceec9734a6f4b57baa986f43ea10b5c9a77420929720124ca0"  # 替换为实际文件路径
    success, prob, label = run_prediction(
        model=model,
        file_path=test_file,
        device=torch.device("cpu")
    )
    
    if success:
        logger.info(f"\n预测结果：")
        logger.info(f"恶意概率: {prob:.4f}")
        logger.info(f"分类结果: {'恶意' if label == 1 else '良性'}")
    else:
        logger.error("预测失败")
  