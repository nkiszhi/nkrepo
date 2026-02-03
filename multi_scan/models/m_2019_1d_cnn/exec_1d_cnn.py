import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from configparser import ConfigParser
from models.m_2019_1d_cnn.feature_extraction.extract_feature import extract_features_1d_cnn, scan_load_samples
from models.m_2019_1d_cnn.one_d_cnn import OneD_CNN

# Load configuration
cp = ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini'))
TRAINING_DATA = cp.get('files', 'training_data')
MODEL_PATH = cp.get('files', 'model_path')

# 统一设置 num_classes
NUM_CLASSES = 2

def run_training():
    base_dir = TRAINING_DATA
    train_samples = scan_load_samples(base_dir)
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # 提取特征
    features, labels = extract_features_1d_cnn(train_samples)
    if features is None or labels is None:
        return

    # 转换为 PyTorch 张量
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    input_channels = 1
    model = OneD_CNN(input_channels, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # 保存模型
    torch.save(model.state_dict(), '1d_cnn_model.pth')
    print('Model saved.')

#
#def run_prediction():
#    sample_path = r"C:\zjp\data\32\2d6ab778e1563ab7fddc9c81649ce1a58c6f27565df877b035c16894f778cc1d"
#    test_samples = scan_load_samples(sample_path)
#    # 提取特征
#    features, labels = extract_features_1d_cnn(test_samples)
#    if features is None or labels is None:
#        return
#
#    # 转换为 PyTorch 张量
#    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
#    labels = torch.tensor(labels, dtype=torch.long)
#
#    # 创建数据集和数据加载器
#    test_dataset = TensorDataset(features, labels)
#    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#    # 加载训练好的模型
#    input_channels = 1
#    model = OneD_CNN(input_channels, NUM_CLASSES)
#    try:
#        model.load_state_dict(torch.load('1d_cnn_model.pth'))
#    except FileNotFoundError:
#        print("Model file not found. Please train the model first.")
#        return
#    model.eval()
#
#    predictions = []
#    with torch.no_grad():
#        for inputs, _ in test_loader:
#            outputs = model(inputs)
#            _, predicted = torch.max(outputs.data, 1)
#            predictions.extend(predicted.cpu().tolist())
#
#    print('Predictions:', predictions)
def run_prediction(file_path):
    """
    运行 1D-CNN 模型的预测流程（安全加载模型，消除 FutureWarning）
    """
    device = 'cpu'  # 强制使用 CPU
    input_channels = 1
    model = OneD_CNN(input_channels, NUM_CLASSES)
    model_path = os.path.join(MODEL_PATH, 'm_2019_1d_cnn', 'saved', '1d_cnn_model.pth')  # 实际模型路径

    # 1. **安全加载模型（关键修正：显式设置 weights_only=True）**
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在！")
        return None
    try:
        # 仅加载模型权重，禁止加载任意对象（遵循 PyTorch 安全建议）
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True  # 显式设置，消除警告并提升安全性
        )
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"模型加载失败：{e}")
        return None
    model.eval().to(device)  # 评估模式 + CPU 设备

    # 2. 文件加载与特征提取（与训练时逻辑一致）
    test_samples = scan_load_samples(file_path)
    if not test_samples:
        print(f"错误：文件 {file_path} 不存在或格式错误！")
        return None

    features, _ = extract_features_1d_cnn(test_samples)
    if features is None or len(features) == 0:
        print("错误：特征提取失败！")
        return None

    # 3. **输入维度调整（确保符合 1D-CNN 输入要求）**
    # 假设 extract_features_1d_cnn 返回形状为 (seq_len,) 的单样本特征
    if len(features.shape) == 1:
        features = features[np.newaxis, np.newaxis, :]  # 添加批次维度和通道维度，变为 (1, 1, seq_len)
    elif len(features.shape) == 2:
        features = features[:, np.newaxis, :]  # 多样本时添加通道维度，变为 (batch_size, 1, seq_len)
    
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    # 4. 模型预测（与训练时逻辑一致）
    with torch.no_grad():
        outputs = model(features_tensor)
        # 计算恶意概率（假设类别 1 为恶意，softmax 输出概率分布）
        probabilities = torch.softmax(outputs, dim=1)
        malicious_prob = probabilities[0, 1].item()  # 提取第一个样本的恶意概率

    return round(malicious_prob, 4)



