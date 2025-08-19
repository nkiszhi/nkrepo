from models.feature_extraction.extract_feature import scan_load_samples
from models.m_2017_malconv.train import *
from models.m_2017_malconv.predict import *


def run_training():
    """
    运行 Malconv 模型的训练流程。
    - 包括特征提取、数据加载、训练和模型保存。
    """
    # 定义路径和参数
    base_dir = "E:\Experimental data\dr_data"  # 样本根目录
    h5_path = "malconv_features.h5"  # HDF5 文件路径
    max_len = 200000  # 最大样本长度
    batch_size = 64  # 批量大小
    epochs = 20  # 训练轮次
    device = 'cuda'  # 运行设备

    # 检查是否需要提取特征
    if not os.path.exists(h5_path):
        print("H5 文件不存在，开始提取特征...")
        samples = scan_load_samples(base_dir)
        save_to_h5(samples, h5_path, max_len=max_len, batch_size=batch_size)
    else:
        print("H5 文件已存在，跳过特征提取。")

    # 加载 H5 数据并划分训练/测试集
    print("加载 H5 数据并划分训练/测试集...")
    data, labels = load_data_from_h5(h5_path)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, val_size=0.1)

    # 初始化模型
    model = Malconv()
    print("开始训练模型...")

    # 训练模型
    train(model, x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=epochs, device=device)
    print("模型训练完成！")


def run_prediction(file_path):
    """
    运行 Malconv 模型的预测流程。
    - 包括模型加载、样本预处理和预测。
    """
    # 定义设备和模型参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Malconv(max_len=200000, win_size=500, vocab_size=256)

    # 模型权重路径
    model_path = './models/m_2017_malconv/saved/malconv_best.pth'

    # 检查模型权重文件
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"模型已加载: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")

    # 样本目录
    sample_folder = file_path

    # 调用预测函数
    print("开始预测...")
    results = predict_from_directory(model, sample_folder, batch_size=64, device=device)

    # 输出预测结果
    print("预测结果：")
    for file_path, score in results:
        print(f"文件: {file_path}, 恶意概率: {score:.4f}")
    return round(score, 4)

if __name__ == '__main__':
    # run_training()
    run_prediction()




