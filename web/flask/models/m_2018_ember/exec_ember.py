import os

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from configparser import ConfigParser
from models.m_2018_ember.ember import Ember
from models.m_2018_ember.feature_extraction.extract_feature import extract_features_ember, scan_load_samples
from models.m_2018_ember.train import train_model

# Load configuration
cp = ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini'))
TRAINING_DATA = cp.get('files', 'training_data')
MODEL_PATH = cp.get('files', 'model_path')


class PEFeatureDataset(Dataset):
    """
    PE文件特征数据集
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate_model(model, test_loader, device="cpu"):
    """
    评估模型
    """
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs.squeeze() > 0.5).cpu().numpy()
            predictions.extend(preds)
            truths.extend(y_batch.numpy())

    accuracy = accuracy_score(truths, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


# 训练流程
def run_training():
    base_dir = TRAINING_DATA
    all_samples = scan_load_samples(base_dir)
    print("Extracting features...")
    all_features, all_labels = extract_features_ember(all_samples)
    print("Combining and splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_dataset = PEFeatureDataset(X_train, y_train)
    test_dataset = PEFeatureDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Initializing model...")
    model = Ember(input_dim=X_train.shape[1], num_trees=10, tree_depth=3, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    print("Training model...")
    train_model(model, train_loader, optimizer, criterion, num_epochs=10)
    print("Evaluating model...")
    evaluate_model(model, test_loader)
    print("Saving model...")
    torch.save(model.state_dict(), "./saved/ember_model.pth")
    print("Training complete!")


def run_prediction():
    """
    预测流程
    """
    model_path = os.path.join(MODEL_PATH, 'm_2018_ember', 'saved', 'ember_model.pth')
    sample_path = r"../yucedata/2dbb5d05211fd4990685d9373c906eafa900b11603d4701c0c347876e820a197"
    print("Loading model...")
    model = Ember(input_dim=2381, num_trees=10, tree_depth=3, output_dim=1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("Extracting sample features...")
    sample_features, _ = extract_features_ember(scan_load_samples(sample_path))
    print("Normalizing sample features...")
    # 加载标准化参数
    scaler = joblib.load('./saved/scaler.pkl')
    sample_features = scaler.transform(sample_features.reshape(1, -1))
    print("Making prediction...")
    with torch.no_grad():
        prediction = model(torch.tensor(sample_features, dtype=torch.float32)).item()
    print(f"Prediction for {sample_path}: {'Malicious' if prediction > 0.5 else 'Benign'}")

def run_prediction(file_path):
    """Ember模型预测流程"""
    # 定义设备和模型参数
    device = 'cpu'  # 强制使用CPU
    model = Ember(input_dim=2381, num_trees=10, tree_depth=3, output_dim=1)
    model_path = os.path.join(MODEL_PATH, 'm_2018_ember', 'saved', 'ember_model.pth') 
    
    # 加载模型权重
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在！")
        return None
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    
    try:
        # 直接传递文件路径字符串，获取文件字节内容
        sample_bytes = scan_load_samples(file_path)
    except FileNotFoundError as e:
        print(f"文件加载失败：{e}")
        return None
    
    # 提取特征
    features, _ = extract_features_ember(sample_bytes)  
    
    # 标准化特征
    scaler = joblib.load(os.path.join(MODEL_PATH, 'm_2018_ember', 'saved', 'scaler.pkl'))
    sample_features = scaler.transform(features.reshape(1, -1))
    
    # 模型预测
    with torch.no_grad():
        inputs = torch.tensor(sample_features, dtype=torch.float32).to(device)
        prediction = model(inputs).item()
        return round(prediction, 4)

if __name__ == "__main__":
    # run_training()
    run_prediction()

