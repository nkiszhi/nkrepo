import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from models.m_2021_malconv2.extract_feature import scan_load_samples, extract_features_malconv
from models.m_2021_malconv2.malconv2 import MalConvGCT

class TrainDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def run_training():
    # 配置信息
    samples_base_dir = "E:\Experimental data\dr_data"
    max_len = 16000000
    out_size = 2
    channels = 128
    window_size = 512
    stride = 64
    embd_size = 8
    epochs = 30
    batch_size = 128
    learning_rate = 0.001
    non_neg = False
    gpus = [0]  # 假设使用第一个 GPU

    # 初始化设备
    device = torch.device(f"cuda:{gpus[0]}" if gpus and torch.cuda.is_available() else "cpu")

    # 加载样本
    samples = scan_load_samples(samples_base_dir)

    # 提取特征
    features, labels = extract_features_malconv(samples, max_len)
#    print(f"成功提取 {len(labels)} 个样本特征")

    # 划分训练集和验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = TrainDataset(train_features, train_labels)
    val_dataset = TrainDataset(val_features, val_labels)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型
    model = MalConvGCT(
        out_size=out_size,
        channels=channels,
        window_size=window_size,
        stride=stride,
        embd_size=embd_size,
        low_mem=False
    ).to(device)

    # 多 GPU 支持
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 模型保存路径
    base_name = f"saved"
    if non_neg:
        base_name = f"NonNeg_{base_name}"
    os.makedirs(base_name, exist_ok=True)
    file_name = os.path.join(base_name, base_name)

    # 日志文件
    with open(f"{file_name}.csv", 'w') as f:
        f.write("epoch,train_acc,train_auc,test_acc,test_auc\n")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_preds = []
        train_truths = []
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 非负训练
            if non_neg:
                for p in model.parameters():
                    p.data.clamp_(0)

            optimizer.step()

            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(torch.softmax(outputs, dim=-1)[:, 1].detach().cpu().numpy())
            train_truths.extend(labels.detach().cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # 计算训练指标
        train_acc = correct / total
        train_auc = roc_auc_score(train_truths, train_preds)

        # 验证
        model.eval()
        val_preds = []
        val_truths = []
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs, _, _ = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(torch.softmax(outputs, dim=-1)[:, 1].detach().cpu().numpy())
                val_truths.extend(labels.detach().cpu().numpy())
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        val_auc = roc_auc_score(val_truths, val_preds)

        # 保存模型
        model_path = os.path.join(base_name, f"malconv_gct_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_auc': train_auc,
            'test_auc': val_auc
        }, model_path)

        # 记录日志
        with open(f"{file_name}.csv", 'a') as f:
            f.write(f"{epoch},{train_acc:.4f},{train_auc:.4f},{val_acc:.4f},{val_auc:.4f}\n")

        print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

    return model

def run_prediction(file_path):
    # 配置日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 配置信息
    samples_base_dir = file_path
    max_len = 16000000
    out_size = 2
    channels = 128
    window_size = 512
    stride = 64
    embd_size = 8
    batch_size = 128
    gpus = [0]  # 假设使用第一个 GPU

    # 初始化设备
    device = torch.device(f"cuda:{gpus[0]}" if gpus and torch.cuda.is_available() else "cpu")

    try:
        # 加载样本
        samples = scan_load_samples(samples_base_dir)
        logging.info(f"加载样本: {samples}")

        # 提取特征
        features, labels = extract_features_malconv(samples, max_len)
        logging.info(f"成功提取 {len(labels)} 个样本特征")

        # 保留所有样本，但在计算指标时过滤 -1
        test_dataset = TrainDataset(features, labels)

        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 初始化模型
        model = MalConvGCT(
            out_size=out_size,
            channels=channels,
            window_size=window_size,
            stride=stride,
            embd_size=embd_size,
            low_mem=False
        ).to(device)

        # 加载训练好的模型
        model_path = "/home/nkamg/nkrepo/zjp/multi_model_detection_system/new_flask/models/m_2021_malconv2/saved/malconv_gct_epoch_29.pth"
        # 指定 map_location=torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_preds = []
        test_truths = []
        correct_test = 0
        total_test = 0
        has_valid_labels = False

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs, _, _ = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # 转换为 numpy 数组
                pred_probs = torch.softmax(outputs, dim=-1)[:, 1].detach().cpu().numpy()
                true_labels = labels.detach().cpu().numpy()

                # 记录所有预测结果
                test_preds.extend(pred_probs)
                test_truths.extend(true_labels)

                # 统计有效样本（标签不为 -1）
                valid_mask = (true_labels != -1)
                correct_test += np.sum(predicted[valid_mask].cpu().numpy() == true_labels[valid_mask])
                total_test += np.sum(valid_mask)
                has_valid_labels = has_valid_labels or np.any(valid_mask)

        # 输出预测结果
        if len(test_preds) == 1:
            logging.info(f"Predicted probability: {test_preds[0]:.4f}")
        else:
            logging.info(f"Predicted probabilities: {test_preds}")

        # 计算指标
        if has_valid_labels:
            test_acc = correct_test / total_test
            if len(np.unique([label for label in test_truths if label != -1])) > 1:
                test_auc = roc_auc_score([label for label in test_truths if label != -1], test_preds)
                logging.info(f"Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")
            else:
                logging.info(f"Test Acc: {test_acc:.4f} | ROC AUC score is not defined as only one class is present in y_true.")
        else:
            logging.info("No valid samples with labels for metric calculation.")

        # 修改返回值
        if len(test_preds) > 0:
            return test_preds[0]
        else:
            return None

    except Exception as e:
        logging.error(f"预测过程中出现错误: {str(e)}")
        return None

# 示例运行
if __name__ == "__main__":
    #run_training()
    run_prediction()

