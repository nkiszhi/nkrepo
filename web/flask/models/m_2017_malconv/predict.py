import os

import numpy as np
import torch

from models.feature_extraction.extract_feature import scan_load_prediction_samples
from models.m_2017_malconv.malconv import Malconv
from models.m_2017_malconv.utils import data_generator, predict_data_loader, preprocess


def predict(model, processed_data, labels, batch_size=64, verbose=1, device='cuda'):
    """
    使用训练好的 PyTorch 模型进行预测。

    参数:
        model (nn.Module): 已训练的 PyTorch 模型。
        processed_data (list): 预处理后的数据列表，每个元素是文件的字节序列张量。
        labels (list): 标签列表（用于占位，无实际用途，可传空列表）。
        batch_size (int): 每批次大小。
        verbose (int): 日志输出级别。
        device (str): 使用的设备 ("cuda" 或 "cpu")。

    返回:
        predictions (list): 每个输入的预测概率值列表。
    """
    # 将模型转移到指定设备
    model = model.to(device)
    model.eval()  # 设置模型为评估模式

    # 转换数据为 NumPy 数组以便索引
    data = torch.stack([x.clone().detach() for x in processed_data])
    labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros(len(processed_data))

    # 创建生成器
    data_gen = data_generator(data.numpy(), labels.numpy(), batch_size, shuffle=False)

    # 计算生成器的总步数
    steps = len(processed_data) // batch_size + int(len(processed_data) % batch_size > 0)

    # 存储预测结果
    predictions = []

    with torch.no_grad():
        for step in range(steps):
            # 从生成器获取一批数据
            batch_data, _ = next(data_gen)
            batch_data = batch_data.to(device)  # 转移到设备

            # 前向传播
            output = model(batch_data).squeeze(1)  # 输出的形状 (batch_size, )
            predictions.extend(output.cpu().numpy())  # 转换为 NumPy 格式并存储

            if verbose > 0:
                print(f"Processed batch: {len(predictions)}/{len(processed_data)}")

    return predictions


def predict_from_directory(model, folder_path, batch_size=64, device='cuda'):
    """
    从文件夹中加载所有样本，并使用训练好的模型进行预测。

    参数:
        model (nn.Module): 已训练的 Malconv 模型。
        folder_path (str): 样本文件所在的目录。
        batch_size (int): 每批次大小。
        device (str): 使用的设备 ("cuda" 或 "cpu")。

    返回:
        results (list of tuple): 文件路径及对应的恶意概率 [(sample_path, score), ...]
    """
    file_list = scan_load_prediction_samples(folder_path)

    if not file_list:
        raise ValueError(f"目录 {folder_path} 中没有发现任何文件！")
    file_list = [str(file_list[0][0])]
    print("正在预处理数据...")
    processed_data, _ = preprocess(file_list, 200000)
    dummy_labels = [0] * len(file_list)  # 使用占位标签

    # 调用预测函数
    predictions = predict(model, processed_data, dummy_labels, batch_size=batch_size, device=device)

    # 整理结果
    results = list(zip(file_list, predictions))

    return results

