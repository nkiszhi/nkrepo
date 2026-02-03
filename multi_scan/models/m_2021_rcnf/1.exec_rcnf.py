import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import logging
from models.m_2021_rcnf.CapsNet import CapsNet
from models.m_2021_rcnf.RCNF import RCNF
from models.m_2021_rcnf.extract_feature import scan_load_samples, extract_features_rcnf

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training():
    """
    无参数训练函数（硬编码路径）
    数据集结构：
    E:\Experimental data\dr_data\
    ├── benign_unpacked\benign\ （良性样本，标签0）
    └── malicious_unpacked\     （恶意样本，标签1）
    """
    # 硬编码路径
    base_dir = r"E:\Experimental data\dr_data"
    model_save_path = "rcnf_model.pth"
    img_size = 224
    n_estimators = 5
    epochs = 3
    batch_size = 8
    val_ratio = 0.2  # 20% 验证集

    # 1. 加载所有样本
    logger.info("Loading samples from hardcoded path...")
    all_samples = scan_load_samples(base_dir)
    if not all_samples:
        raise FileNotFoundError("No samples found in hardcoded directory")

    logger.info(f"Loaded {len(all_samples)} samples")

    # 2. 划分训练集/验证集（随机划分，不进行分层）
    np.random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - val_ratio))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # 3. 特征提取
    logger.info("Extracting features...")
    train_features, train_labels = extract_features_rcnf(train_samples, img_size)
    val_features, val_labels = extract_features_rcnf(val_samples, img_size)
    if train_features is None or val_features is None:
        raise RuntimeError("Feature extraction failed")

    # 4. 训练模型
    logger.info("Training RCNF...")
    rcnf = RCNF(base_model_class=CapsNet, n_estimators=n_estimators, num_classes=2)
    train_dataset = TensorDataset(train_features, torch.LongTensor(train_labels))
    val_dataset = TensorDataset(val_features, torch.LongTensor(val_labels))
    rcnf.fit(trainset=train_dataset, valset=val_dataset, epochs=epochs, batch_size=batch_size)

    # 5. 保存模型
    torch.save(rcnf.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


def run_prediction():

    test_samples_dir = r"/home/user/MCDM/csdata/be/a9afc207bdf60e25a9a395b511dff7468ba1d073f6a25a73223c7ba6725694a8"
    model_path = "rcnf_model.pth"
    img_size = 224

    # 1. 加载测试样本
    logger.info("Loading test sample...")
    test_samples = scan_load_samples(test_samples_dir)
    if not test_samples:
        raise FileNotFoundError("Test sample not found")

    logger.info(f"Test sample: {test_samples[0][0]}")

    # 2. 特征提取
    logger.info("Extracting test features...")
    test_features, _ = extract_features_rcnf(test_samples, img_size)
    if test_features is None:
        raise RuntimeError("Test feature extraction failed")

    # 3. 加载模型
    logger.info("Loading model...")
    rcnf = RCNF(base_model_class=CapsNet, n_estimators=5, num_classes=2)
    rcnf.load_state_dict(torch.load(model_path))

    # 4. 预测
    logger.info("Predicting...")
    dataset = torch.utils.data.TensorDataset(test_features)
    prediction = rcnf.predict(dataset)
    result_mapping = {0: '良性', 1: '恶意'}
    result = result_mapping[prediction[0]]
    logger.info(f"预测结果: {result}")

if __name__ == '__main__':
    #run_training()
    run_prediction()