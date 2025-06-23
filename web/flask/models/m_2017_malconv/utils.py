import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

def limit_gpu_memory(per):
    """
    限制 GPU 显存的使用比例。
    参数:
        per (float): GPU 显存使用比例 (0.0 ~ 1.0)。
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(per)
        print(f"GPU memory usage limited to {per * 100:.1f}%")
    else:
        print("No GPU available. Skipping GPU memory configuration.")


def preprocess(fn_list, max_len):
    """
    返回处理后的数据（Tensor）和原始文件长度（list）。
    参数:
        fn_list (list of str): 文件路径列表。
        max_len (int): 用于填充和截断的最大长度。
    返回:
        seq (torch.Tensor): 填充和截断后的序列张量。
        len_list (list of int): 原始文件长度列表。
    """
    corpus = []  # 用于存储每个文件的字节内容
    len_list = []  # 用于记录每个文件的原始长度

    for fn in fn_list:
        if not os.path.isfile(fn):
            print(f"{fn} 文件不存在")  # 如果文件不存在，打印提示
        else:
            with open(fn, 'rb') as f:
                # 读取文件内容并转化为 PyTorch 的张量
                content = torch.tensor(list(f.read()), dtype=torch.uint8)
                corpus.append(content)  # 将字节内容添加到语料库
                len_list.append(len(content))  # 记录文件的原始长度

    # 使用 pad_sequence 对序列进行填充
    padded_corpus = pad_sequence(
        corpus,
        batch_first=True,  # 确保输出的维度是 (batch_size, max_len)
        padding_value=0  # 填充值为 0
    )
    # 如果序列长度超过 max_len，进行截断
    padded_corpus = padded_corpus[:, :max_len]

    # 如果序列长度小于 max_len，再次填充保证长度一致
    if padded_corpus.size(1) < max_len:
        padded_corpus = torch.nn.functional.pad(
            padded_corpus,
            (0, max_len - padded_corpus.size(1)),  # 填充到 max_len
            value=0  # 填充值为 0
        )

    return padded_corpus, len_list  # 返回处理后的张量和原始长度列表


def load_samples(base_dir):
    """
    加载样本路径和标签。
    参数:
        base_dir (str): 样本存放的基础目录。
    返回:
        list: 包含 (文件路径, 标签) 的元组列表。
    """
    def get_sample_paths(the_base_dir, label):
        sample_list = []
        for root, _, files in os.walk(the_base_dir):
            for file in files:
                sample_list.append((os.path.join(root, file), label))
        return sample_list

    # 良性样本
    benign_dir = os.path.join(base_dir, "benign_unpacked/benign")
    benign_samples = get_sample_paths(benign_dir, label=0)

    # 恶性样本
    malicious_samples = []
    malicious_dir = os.path.join(base_dir, "malicious_unpacked")
    for year in os.listdir(malicious_dir):
        year_path = os.path.join(malicious_dir, year)
        if os.path.isdir(year_path):
            for family in os.listdir(year_path):
                family_path = os.path.join(year_path, family)
                if os.path.isdir(family_path):
                    malicious_samples.extend(get_sample_paths(family_path, label=1))

    all_samples = benign_samples + malicious_samples
    # 转换为 NumPy 数组并打乱
    all_samples_array = np.array(all_samples, dtype=object)
    np.random.shuffle(all_samples_array)
    # 转回列表
    return all_samples_array.tolist()

def save_to_h5(samples, h5_path, max_len=200000, batch_size=64):
    """
    将特征和标签保存到 HDF5 文件中。
    """
    with h5py.File(h5_path, 'w') as h5f:
        data_shape = (len(samples), max_len)
        labels_shape = (len(samples),)
        data_dset = h5f.create_dataset("data", shape=data_shape, dtype='uint8')
        labels_dset = h5f.create_dataset("labels", shape=labels_shape, dtype='uint8')

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_files = [file_path for file_path, _ in batch]
            batch_labels = [label for _, label in batch]
            batch_features, _ = preprocess(batch_files, max_len)
            data_dset[i:i + len(batch_features)] = batch_features.numpy()
            labels_dset[i:i + len(batch_labels)] = batch_labels
            print(f"已处理 {i + len(batch_features)}/{len(samples)} 样本")


def load_data_from_h5(h5_path):
    """
    从 HDF5 文件中加载特征数据和标签。

    参数:
        h5_path (str): HDF5 文件的路径。

    返回:
        data (np.ndarray): 加载的特征数据。
        labels (np.ndarray): 加载的标签数据。
    """
    with h5py.File(h5_path, 'r') as h5f:
        # 读取特征数据
        data = h5f['data'][:]
        # 读取标签数据
        labels = h5f['labels'][:]

    return data, labels

def train_test_split(data, label, val_size=0.1):
    """
    数据划分为训练集和测试集。
    参数:
        data (np.ndarray): 输入数据。
        label (np.ndarray): 标签。
        val_size (float): 测试集所占比例。
    返回:
        x_train, x_test, y_train, y_test: 划分后的数据和标签。
    """
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    split = int(len(data) * val_size)
    x_train, x_test = data[idx[split:]], data[idx[:split]]
    y_train, y_test = label[idx[split:]], label[idx[:split]]
    return x_train, x_test, y_train, y_test


def data_generator(data, labels, batch_size=64, shuffle=True):
    """
    数据生成器，用于从已预处理的数据和标签中批量生成训练数据。
    参数:
        data (np.ndarray): 已预处理的特征数据。
        labels (np.ndarray): 标签。
        batch_size (int): 每批次大小。
        shuffle (bool): 是否打乱数据顺序。
    返回:
        Generator: 每次生成 (batch_data, batch_labels)。
    """
    idx = np.arange(len(data))
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(data), batch_size):
            batch_indices = idx[i:i + batch_size]
            batch_data = torch.tensor(data[batch_indices], dtype=torch.float32)
            batch_labels = torch.tensor(labels[batch_indices], dtype=torch.float32)
            yield batch_data, batch_labels


def dataset_loader(base_dir, h5_path, max_len, batch_size):
    """
    加载数据集并预处理为训练和测试数据。

    参数:
        base_dir (str): 样本存放的基础目录。
            - 良性样本应位于 "base_dir/benign_unpacked/benign/"。
            - 恶性样本应位于 "base_dir/malicious_unpacked/年份/恶意家族分类/"。
        h5_path (str): 输出 HDF5 文件的路径，用于存储或加载提取的特征。
        max_len (int): 每个样本的最大字节序列长度。
        batch_size (int): 特征提取和存储时使用的批量大小。

    返回:
        tuple: 训练集和测试集，包括 (x_train, x_test, y_train, y_test)。
            - x_train (np.ndarray): 训练集特征。
            - x_test (np.ndarray): 测试集特征。
            - y_train (np.ndarray): 训练集标签。
            - y_test (np.ndarray): 测试集标签。

    功能描述:
        1. 如果指定的 HDF5 文件已存在，直接加载特征和标签。
        2. 如果 HDF5 文件不存在，则遍历样本目录，提取样本特征并保存到 HDF5 文件中。
        3. 从 HDF5 文件中加载特征和标签。
        4. 根据指定比例将数据划分为训练集和测试集。

    示例:
        base_dir = "./sample_data"
        h5_path = "./features.h5"
        max_len = 200000
        batch_size = 64
        x_train, x_test, y_train, y_test = dataset_loader(base_dir, h5_path, max_len, batch_size)
    """
    # 检查是否已存在提取后的 HDF5 文件
    if os.path.exists(h5_path):
        print(f"{h5_path} 已存在，跳过特征提取。")
    else:
        # 加载样本路径和标签
        samples = load_samples(base_dir)
        # 提取特征并保存到 HDF5 文件
        save_to_h5(samples, h5_path, max_len=max_len, batch_size=batch_size)

    # 从 HDF5 文件中加载特征和标签
    data, labels = load_data_from_h5(h5_path)

    # 划分数据集为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data, labels)

    return x_train, x_test, y_train, y_test

def predict_data_loader(folder_path, max_len=200000):
    """
    从文件夹中加载并预处理待预测数据。
    参数:
        folder_path (str): 待预测的文件夹路径。
        max_len (int): 每个文件的最大字节序列长度。
    返回:
        data (torch.Tensor): 预处理后的张量数据。
        labels (list): 占位标签，用于生成器。
        file_list (list): 加载的文件路径列表。
    """
    # 遍历文件夹，获取所有文件路径
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    if not file_list:
        raise ValueError(f"文件夹 {folder_path} 中没有发现任何文件！")

    # 加载和预处理数据
    data, _ = preprocess(file_list, max_len)
    labels = [0] * len(file_list)  # 使用占位标签

    return data, labels, file_list


class Logger:
    """
    日志记录器，用于保存和输出训练过程中的文件信息。
    """
    def __init__(self):
        self.fn = []       # 文件名
        self.len = []      # 文件原始长度
        self.pad_len = []  # 填充长度
        self.loss = []     # 损失值
        self.pred = []     # 预测值
        self.org = []      # 原始分数

    def write(self, fn, org_score, file_len, pad_len, loss, pred):
        """
        记录单个文件的日志信息。
        参数:
            fn (str): 文件名。
            org_score (float): 原始分数。
            file_len (int): 文件长度。
            pad_len (int): 填充长度。
            loss (float): 损失值。
            pred (float): 预测分数。
        """
        self.fn.append(fn.split('/')[-1])
        self.org.append(org_score)
        self.len.append(file_len)
        self.pad_len.append(pad_len)
        self.loss.append(loss)
        self.pred.append(pred)

        print('\nFILE:', fn)
        if pad_len > 0:
            print('\tfile length:', file_len)
            print('\tpad length:', pad_len)
            print('\tloss:', loss)
            print('\tscore:', pred)
        else:
            print('\tfile length:', file_len, ', Exceed max length! Ignored!')
        print('\toriginal score:', org_score)

    def save(self, path):
        """
        将日志保存到 CSV 文件中。
        参数:
            path (str): 文件保存路径。
        """
        d = {
            'filename': self.fn,
            'original score': self.org,
            'file length': self.len,
            'pad length': self.pad_len,
            'loss': self.loss,
            'predict score': self.pred
        }
        df = pd.DataFrame(data=d)
        df.to_csv(path, index=False, columns=[
            'filename', 'original score', 'file length',
            'pad length', 'loss', 'predict score'
        ])
        print(f'\nLog saved to "{path}"\n')