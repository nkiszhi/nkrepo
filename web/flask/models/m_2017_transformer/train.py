import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformer import TransformerClassifier

class MalwareDataset(Dataset):
    """定义用于加载恶意软件数据的 Dataset"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# -------------------------------------
# 模型训练
# -------------------------------------

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, correct, total = 0, 0, 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # 验证阶段
        model.eval()
        valid_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, labels in valid_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        valid_acc = 100. * correct / total
        print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {valid_acc:.2f}%\n")


# -------------------------------------
# 数据生成与加载
# -------------------------------------

def generate_mock_data(num_samples=1000, seq_len=512, num_classes=2):
    X = np.random.randint(0, 256, (num_samples, seq_len))  # 模拟字节序列
    y = np.random.randint(0, num_classes, num_samples)  # 模拟标签
    return X, y


# 超参数定义
SEQ_LEN = 512
EMBED_DIM = 128
NUM_HEADS = 8
FF_DIM = 256
NUM_LAYERS = 4
NUM_CLASSES = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据生成
X_train, y_train = generate_mock_data(num_samples=5000, seq_len=SEQ_LEN, num_classes=NUM_CLASSES)
X_valid, y_valid = generate_mock_data(num_samples=1000, seq_len=SEQ_LEN, num_classes=NUM_CLASSES)

train_dataset = MalwareDataset(X_train, y_train)
valid_dataset = MalwareDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型初始化
model = TransformerClassifier(
    seq_len=SEQ_LEN,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 模型训练
train_model(model, train_loader, valid_loader, criterion, optimizer, EPOCHS, DEVICE)
