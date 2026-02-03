import os
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from configparser import ConfigParser
from models.m_2017_transformer.transformer import TransformerClassifier
from models.m_2017_transformer.feature_extraction.extract_feature import extract_features_transforms, scan_load_samples

# Load configuration
cp = ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini'))
TRAINING_DATA = cp.get('files', 'training_data')
MODEL_PATH = cp.get('files', 'model_path')


def run_training():
    train_samples = TRAINING_DATA

    # 定义训练样本列表
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # 提取训练数据特征并创建数据集和数据加载器
    train_dataset = extract_features_transforms(scan_load_samples(train_samples))
    if train_dataset is None:
        return None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = TransformerClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # 保存模型
    torch.save(model.state_dict(), 'transformer_classifier.pth')
    print('Model saved.')
    return model


#def run_prediction():
#    # 定义测试样本列表
#    test_samples = r"../../test_samples/2de9cf1c749c7dbb60fa6dc23982c075373ca37879683143d15bc5a6ac9db34b"
#
#    # 加载训练好的模型
#    model = TransformerClassifier()
#    try:
#        model.load_state_dict(torch.load('transformer_classifier.pth'))
#    except FileNotFoundError:
#        print("Model file not found. Please train the model first.")
#        return None
#    model.eval()
#
#    # 提取测试数据特征并创建数据集和数据加载器
#    test_dataset = extract_features_transforms(scan_load_samples(test_samples))
#    if test_dataset is None:
#        return None
#    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#    predictions = []
#    with torch.no_grad():
#        for inputs, _ in test_loader:
#            outputs = model(inputs)
#            _, predicted = torch.max(outputs.data, 1)
#            predictions.extend(predicted.cpu().tolist())
#
#    return predictions

def run_prediction(file_path):
    """Transformer 模型预测函数（单文件版本）"""
    device = 'cpu'  # 强制使用 CPU
    model = TransformerClassifier()
    model_path = os.path.join(MODEL_PATH, 'm_2017_transformer', 'saved', 'transformer_classifier.pth')
    
    # 安全加载模型（修正 FutureWarning）
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在！")
        return None
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True  # 关键修正：启用安全加载
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    # 提取单个文件特征（修正路径传递错误）
    try:
        sample_content = scan_load_samples(file_path)  # 接收单个文件路径字符串
    except FileNotFoundError as e:
        print(f"文件加载错误: {e}")
        return None
    
    test_dataset = extract_features_transforms(sample_content)
    if test_dataset is None:
        print("特征提取失败")
        return None
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)  # 移动到 CPU
            outputs = model(inputs)
            # 假设输出为 logits，计算恶意概率（假设类别 1 是恶意）
            probability = torch.softmax(outputs, dim=1)[0][1].item()
            return round(probability, 4)
if __name__ == "__main__":
    # 训练模型
    # trained_model = run_training()

    # 进行预测
    # if trained_model:
    predictions = run_prediction()
    print('Predictions:', predictions)


