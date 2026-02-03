import torch
import torch.nn as nn
import torchvision.models as models

# Inception V3模型
# InceptionV3Model 类的修改
class InceptionV3Model(nn.Module):
    def __init__(self, num_classes=25):
        super(InceptionV3Model, self).__init__()
        self.inception_v3 = models.inception_v3(weights=None, init_weights=True)
        num_ftrs = self.inception_v3.fc.in_features
        self.inception_v3.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # 禁用辅助分类器
        self.inception_v3.aux_logits = False
        return self.inception_v3(x)