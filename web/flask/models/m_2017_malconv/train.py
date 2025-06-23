import torch
import torch.nn as nn
import torch.optim as optim
from models.m_2017_malconv.utils import *
from models.m_2017_malconv.malconv import Malconv
import os


def train(model, train_data, train_labels, test_data, test_labels,
          batch_size=64, epochs=100, save_path='./saved/', save_best=True, device='cuda'):
    """
    使用 PyTorch 训练 Malconv 模型
    参数:
        model (nn.Module): Malconv 模型。
        train_data (list): 训练数据集。
        train_labels (list): 训练数据标签。
        test_data (list): 测试数据集。
        test_labels (list): 测试数据标签。
        max_len (int): 最大输入长度。
        batch_size (int): 每批次大小。
        epochs (int): 训练轮次。
        save_path (str): 模型保存路径。
        save_best (bool): 是否仅保存最优模型。
        device (str): 使用的设备 ("cuda" 或 "cpu")。
    返回:
        history (dict): 记录训练损失和验证损失的字典。
    """
    # 将模型转移到指定设备
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建数据生成器
    train_generator = data_generator(train_data, train_labels, batch_size, shuffle=True)
    test_generator = data_generator(test_data, test_labels, batch_size, shuffle=False)

    # 训练和验证的损失记录
    history = {'train_loss': [], 'val_loss': []}

    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    best_val_loss = float('inf')

    # 训练过程
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = len(train_data) // batch_size + 1
        for step in range(train_steps):
            # 从生成器中获取数据
            data, labels = next(train_generator)
            data, labels = data.to(device), labels.to(device)

            # 前向传播
            outputs = model(data).squeeze(1)

            # 计算损失
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证过程
        model.eval()
        val_loss = 0.0
        val_steps = len(test_data) // batch_size + 1
        with torch.no_grad():
            for step in range(val_steps):
                # 从生成器中获取数据
                data, labels = next(test_generator)
                data, labels = data.to(device), labels.to(device)
                outputs = model(data).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # 记录损失
        train_loss /= train_steps
        val_loss /= val_steps
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最优模型
        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'malconv_best.pth'))
            print("Best model saved!")

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(save_path, 'malconv_last.pth'))

    return history