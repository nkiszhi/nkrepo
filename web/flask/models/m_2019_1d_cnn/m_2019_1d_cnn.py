import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.m_2019_1d_cnn.one_d_cnn import OneD_CNN

def predict_single_file(model, file_path, seq_len=512, device="cuda"):
    """
    1D-CNN单文件预测核心函数
    返回格式：(文件路径, 恶意概率, 预测结果)
    -1表示预测失败
    """
    try:
        # 输入验证
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError("空文件")

        # 自动设备检测
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 特征提取
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        
        byte_values = list(raw_bytes)[:seq_len]  # 截断
        if len(byte_values) < seq_len:
            byte_values += [0] * (seq_len - len(byte_values))  # 填充

        # 转换为张量
        features = torch.tensor(
            [byte_values], 
            dtype=torch.float32
        ).unsqueeze(1).to(device)  # (batch, channel, length)

        # 执行预测
        model.eval()
        with torch.no_grad():
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            malicious_prob = probs[0][1].item()

        return (file_path, malicious_prob, 1 if malicious_prob >= 0.5 else 0)

    except Exception as e:
        print(f"[错误] 预测失败：{str(e)}")
        return (file_path, 0.0, -1)

def run_prediction(model, file_path, device, scaler=None):
    """
    1D-CNN预测入口函数
    返回格式：(文件路径, 恶意概率, 预测结果)
    """
    try:
        # 设备设置
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        # 执行预测
        _, malicious_prob, label = predict_single_file(model, file_path, device=device)
        
        # 处理预测失败的情况
        if label == -1:
            return (file_path, 0.0, "错误")

        # 结果格式化
        result = (
            file_path,
            malicious_prob,
            1 if malicious_prob >= 0.5 else 0
        )
        
        return result

    except Exception as e:
        print(f"[严重错误] 预测流程异常：{str(e)}")
        return (file_path, 0.0, -1)
        
if __name__ == "__main__":
    # 示例使用
    test_file = r"../yucedata/2d75574a5c44ca27f7d475c179c3b32dfb4a8afa0f1dbc269fe1067bf1aec733"
    run_prediction(test_file)