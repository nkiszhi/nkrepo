import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleLayer(nn.Module):
    """胶囊层（动态路由实现）"""

    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, num_iterations=3):
        super().__init__()
        # 正确维度：[num_route_nodes, in_channels, num_capsules, out_channels]
        self.route_weights = nn.Parameter(
            torch.randn(num_route_nodes, in_channels, num_capsules, out_channels)
        )
        self.num_iterations = num_iterations

    def squash(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return (norm ** 2 / (1 + norm ** 2)) * (x / norm.clamp(min=1e-8))

    def forward(self, x):
        batch_size = x.size(0)
        # x: [b, route_nodes, in_channels]
        # 使用爱因斯坦求和确保维度匹配
        priors = torch.einsum('brc,rcod->brod', x, self.route_weights)  # [b, route, caps, out_ch]

        # 初始化 logits: [batch_size, num_route_nodes, num_capsules]
        logits = torch.zeros(batch_size, priors.size(1), priors.size(2), device=x.device)

        for _ in range(self.num_iterations):
            probs = F.softmax(logits, dim=1)  # 沿路由节点维度（dim=1）softmax
            # 加权求和并激活
            outputs = self.squash((probs.unsqueeze(-1) * priors).sum(dim=1))  # [b, caps, out_ch]

            if _ != self.num_iterations - 1:
                # 计算对数概率更新
                delta_logits = (priors * outputs.unsqueeze(1)).sum(dim=-1)  # [b, route, caps]
                logits += delta_logits
        return outputs


class CapsNet(nn.Module):
    """带分类器的 CapsNet 模型"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        self.capsule_layer = CapsuleLayer(
            num_capsules=25,
            num_route_nodes=3136,  # 56x56=3136
            in_channels=64,
            out_channels=8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),  # [b, 25, 8] → [b, 200]
            nn.Linear(25 * 8, num_classes)
        )

    def forward(self, x):
        """
        输入: [b, 1, 224, 224] 或 [1, 224, 224]（自动补全批次维度）
        输出: [b, num_classes]
        """
        # 确保输入为张量
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be Tensor, got {type(x)}")
        # 自动补全批次维度
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [H, W] → [1, 1, H, W]

        x = self.conv_blocks(x)  # [b, 64, 56, 56]
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 64)  # [b, 3136, 64]
        x = self.capsule_layer(x)  # [b, 25, 8]
        x = self.classifier(x)  # [b, num_classes]
        return x

    def fit(self, train_loader, epochs, val_loader):
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        best_weights = None

        for epoch in range(epochs):
            self.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            self.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = correct / total
                if acc > best_acc:
                    best_acc = acc
                    best_weights = copy.deepcopy(self.state_dict())
        if best_weights:
            self.load_state_dict(best_weights)