import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import torchvision.models as models
from models.m_2020_inceptionv3.extract_feature import extract_features_rcnf, scan_load_samples, scan_load_prediction_samples, \
    extract_features_v3
from models.m_2020_inceptionv3.InceptionV3 import InceptionV3Model

# 训练函数
def run_training():
    malimg_dataset_path = r"E:\Experimental data\dr_data"
    features, labels = extract_features_v3(scan_load_samples(malimg_dataset_path))

    if features is None or labels is None:
        print("No valid training data. Skipping training.")
        return

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = InceptionV3Model()
    # 去掉辅助分类器
    if hasattr(model.inception_v3, 'AuxLogits'):
        model.inception_v3.AuxLogits = None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20  # 增加训练轮数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 只取主分类器的输出
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # 保存模型
    torch.save(model.state_dict(), 'inceptionv3_malware_model.pth')
    print('Model saved.')


# 预测函数
def run_prediction(file_path):
    malimg_dataset_path = file_path
    features, _ = extract_features_v3(scan_load_prediction_samples(malimg_dataset_path))

    if features is None:
        print("No valid test data. Skipping prediction.")
        return

    # 创建数据集和数据加载器
    test_dataset = TensorDataset(features)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = InceptionV3Model()
    try:
        state_dict = torch.load('./models/m_2020_inceptionv3/saved/inceptionv3_malware_model.pth', weights_only=True,map_location=torch.device('cpu'))
        # 过滤掉辅助分类器的权重
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'AuxLogits' not in k}
        model.load_state_dict(filtered_state_dict, strict=False)
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return

    # 去掉辅助分类器
    if hasattr(model.inception_v3, 'AuxLogits'):
        model.inception_v3.AuxLogits = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images[0].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 只取主分类器的输出
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().tolist())

    if predictions:
        return np.mean(predictions)  # 返回平均得分
    else:
        return None


# 预测函数
def run_training():
    malimg_dataset_path = r"E:\Experimental data\dr_data"
    features, labels = extract_features_rcnf(scan_load_samples(malimg_dataset_path))

    if features is None or labels is None:
        print("No valid training data. Skipping training.")
        return

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = InceptionV3Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20  # 增加训练轮数
    device = torch.device("cpu")  # 强制使用CPU
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # 保存模型
    torch.save(model.state_dict(), 'inceptionv3_malware_model.pth')
    print('Model saved.')




if __name__ == "__main__":
    file_path = "/home/user/MCDM/csdata/be/a9aff4cf06a427d9750540f25793815095c1129a2a2c001145cf603479122edd"
    run_prediction(file_path)


