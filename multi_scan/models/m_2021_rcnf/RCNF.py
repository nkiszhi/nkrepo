import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import copy
import numpy as np


class RCNF(nn.Module):
    def __init__(self, base_model_class, n_estimators, num_classes):
        """
        随机胶囊网络森林（RCNF）
        :param base_model_class: 基模型类（如 CapsNet）
        :param n_estimators: 基模型数量
        :param num_classes: 分类任务类别数
        """
        super().__init__()
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.num_classes = num_classes
        self.models = nn.ModuleList([base_model_class(num_classes=num_classes) for _ in range(n_estimators)])
        self.model_weights = []  # 保存每个基模型的最佳权重

    def resample(self, dataset):
        """自助重采样"""
        indices = torch.randint(0, len(dataset), (len(dataset),), dtype=torch.long)
        return Subset(dataset, indices)

    def fit(self, trainset, valset, epochs, batch_size=32):
        """
        训练 RCNF 集成模型
        :param trainset: 训练数据集
        :param valset: 验证数据集
        :param epochs: 训练轮次
        :param batch_size: 批量大小
        """
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        self.model_weights.clear()  # 清空旧权重
        for i in range(self.n_estimators):
            print(f"Training estimator {i + 1}/{self.n_estimators}")
            bs_trainset = self.resample(trainset)
            train_loader = DataLoader(bs_trainset, batch_size=batch_size, shuffle=True)
            model = self.models[i]
            model.fit(train_loader, epochs, val_loader)
            self.model_weights.append(copy.deepcopy(model.state_dict()))  # 保存最佳权重

    def predict(self, testset):
        """处理 TensorDataset 元组输入（最终修复）"""
        if not isinstance(testset, torch.utils.data.TensorDataset):
            raise ValueError("testset must be TensorDataset")

        # 使用自定义 collate 函数解包元组
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch[0][0]  # 解包元组中的张量
        )

        total_preds = np.zeros((len(testset), self.num_classes))

        for model in self.models:
            model.eval()
            with torch.no_grad():
                for inputs in test_loader:  # 直接获取张量
                    if not isinstance(inputs, torch.Tensor):
                        raise TypeError(f"Input is {type(inputs)}, expected Tensor")
                    outputs = model(inputs)
                    total_preds += outputs.cpu().numpy()

        return np.argmax(total_preds / self.n_estimators, axis=1)

    def save(self, path):
        """保存模型权重"""
        torch.save({
            "n_estimators": self.n_estimators,
            "num_classes": self.num_classes,
            "model_weights": self.model_weights
        }, path)

    @classmethod
    def load(cls, path, base_model_class):
        """加载预训练模型"""
        checkpoint = torch.load(path, map_location="cpu")
        rcnf = cls(
            base_model_class=base_model_class,
            n_estimators=checkpoint["n_estimators"],
            num_classes=checkpoint["num_classes"]
        )
        rcnf.model_weights = checkpoint["model_weights"]
        # 加载权重到模型实例
        for i in range(rcnf.n_estimators):
            rcnf.models[i].load_state_dict(rcnf.model_weights[i])
        return rcnf