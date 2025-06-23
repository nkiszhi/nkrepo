import torch
import torch.nn as nn
import torch.nn.functional as F


class Ember(nn.Module):
    """
    实现LightGBM的EMBER模型。
    """

    def __init__(self, input_dim=2381, num_trees=10, tree_depth=3, output_dim=1):
        """
        初始化 Ember 模型。

        参数：
        ----------
        input_dim : int
            输入特征的维度（默认为 EMBER 特征维度 2381）。
        num_trees : int
            模拟树的数量。
        tree_depth : int
            每棵树的深度。
        output_dim : int
            输出维度（二分类任务默认为 1）。
        """
        super(Ember, self).__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.output_dim = output_dim

        # 树的权重和偏置
        self.trees = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 2 ** (tree_depth - 1)),  # 树的第一级
                    nn.ReLU(),
                    nn.Linear(2 ** (tree_depth - 1), output_dim),  # 输出层
                )
                for _ in range(num_trees)
            ]
        )

        # 全局线性权重，用于合并所有树的输出
        self.global_weights = nn.Linear(num_trees * output_dim, output_dim)

    def forward(self, x):
        """
        前向传播。

        参数：
        ----------
        x : torch.Tensor
            输入特征向量，形状为 (batch_size, input_dim)。

        返回：
        ----------
        torch.Tensor
            模型的预测值，形状为 (batch_size, output_dim)。
        """
        tree_outputs = []

        # 通过每棵树计算输出
        for tree in self.trees:
            tree_out = tree(x)
            tree_outputs.append(tree_out)

        # 合并所有树的输出
        all_tree_outputs = torch.cat(tree_outputs, dim=1)  # 形状: (batch_size, num_trees * output_dim)

        # 全局权重组合
        final_output = self.global_weights(all_tree_outputs)  # 形状: (batch_size, output_dim)

        # 二分类概率输出
        return torch.sigmoid(final_output)