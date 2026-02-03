import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def train(
    model, train_loader, test_loader, device, epochs, learning_rate, checkpoint_dir, use_low_mem=False
):
    """
    训练 MalConvGCT 模型。

    参数:
        model (torch.nn.Module): 需要训练的 MalConvGCT 模型。
        train_loader (DataLoader): 训练数据加载器。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 设备 (CPU/GPU)。
        epochs (int): 训练的总轮数。
        learning_rate (float): 优化器学习率。
        checkpoint_dir (str): 保存模型检查点的路径。
        use_low_mem (bool): 是否使用低内存模式。
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 如果使用低内存模式，启用模型的低内存标志
    if use_low_mem:
        model.low_mem = True

    # 开始训练
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_truths = [], []

        print(f"Epoch {epoch + 1}/{epochs} - Training")

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            outputs, _, _ = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            # 收集预测和真实值
            preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()[:, 1]
            train_preds.extend(preds)
            train_truths.extend(labels.cpu().numpy())

        train_auc = roc_auc_score(train_truths, train_preds)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {running_loss:.4f}, Train AUC: {train_auc:.4f}")

        # 验证阶段
        model.eval()
        test_preds, test_truths = [], []

        with torch.no_grad():
            print(f"Epoch {epoch + 1}/{epochs} - Testing")
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, _, _ = model(inputs)

                preds = torch.softmax(outputs, dim=1).cpu().numpy()[:, 1]
                test_preds.extend(preds)
                test_truths.extend(labels.cpu().numpy())

        test_auc = roc_auc_score(test_truths, test_preds)
        print(f"Epoch {epoch + 1}/{epochs} - Test AUC: {test_auc:.4f}")

        # 保存模型检查点
        checkpoint_path = f"{checkpoint_dir}/epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_auc": train_auc,
                "test_auc": test_auc,
            },
            checkpoint_path,
        )
        print(f"Model checkpoint saved to {checkpoint_path}")
