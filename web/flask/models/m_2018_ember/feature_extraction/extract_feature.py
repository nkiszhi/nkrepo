import hashlib
import json
import os
import re
from collections import Counter
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import TensorDataset
# from torch_geometric.data import Data
import h5py
import numpy as np
import pefile
import torch
# from binaryninja import BinaryViewType, SymbolType, MediumLevelILOperation
from sklearn.feature_extraction import FeatureHasher
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


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

# 1. 扫描加载样本路径和标签
def scan_load_samples(base_dir):
    """
    加载样本所在路径和标签。
    eg:[(url1,0),(url2,1),...]
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

    if os.path.isfile(base_dir):
        # 如果传入的是单个文件，将标签设为 -1 表示未知
        return [(base_dir, -1)]
    # 良性样本
    benign_dir = os.path.join(base_dir, "benign_unpacked/benign") # 良性样本所在子目录
    benign_samples = get_sample_paths(benign_dir, label=0)

    # 恶性样本
    malicious_dir = os.path.join(base_dir, "malicious_unpacked") # 恶意样本所在子目录
    malicious_samples = get_sample_paths(malicious_dir, label=1)
    all_samples = benign_samples + malicious_samples
    # 转换为 NumPy 数组并打乱
    all_samples_array = np.array(all_samples, dtype=object)
    np.random.shuffle(all_samples_array)
    # 转回列表
    return all_samples_array.tolist()

# 2. 提取特征，并确保与标签对应
'''
malconv/malconv2 字节序列
'''
def extract_features_malconv(samples_dir, max_len):
    """
    malconv/malconv2都适用
    提取PE文件的字节特征，填充或截断为固定长度。
    参数:
        samples_dir (list): 文件路径和标签元组的列表。
        max_len (int): 最大长度，超过此长度的文件将被截断。
    返回:
        features (Tensor): 提取并填充后的特征张量 (batch_size, max_len)
        labels (Tensor): 对应的标签张量 (batch_size,)
    """
    corpus = []  # 用于存储每个文件的字节内容
    labels = []  # 用于存储每个文件的标签
    for sample, label in samples_dir:
        if not os.path.isfile(sample):
            print(f"{sample} 文件不存在")  # 如果文件不存在，打印提示
        else:
            with open(sample, 'rb') as f:
                # 读取文件内容并转化为 PyTorch 的张量
                content = torch.tensor(list(f.read()), dtype=torch.uint8)
                corpus.append(content)  # 将字节内容添加到语料库
                labels.append(label)  # 将标签添加到标签列表

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

    return padded_corpus, torch.tensor(labels)  # 返回特征和标签

'''
ember 所包含的特征类型
进行了pefile的重写
'''

'''
基类
'''
class FeatureType:
    """特征类型的基类，每种特征都可以从此类继承"""
    name = ''
    dim = 0  # 特征向量的维度

    def __repr__(self):
        return f'{self.name}({self.dim})'

    def raw_features(self, bytez, pe_binary):
        """生成文件的原始特征表示（JSON 格式）"""
        raise NotImplementedError

    def process_raw_features(self, raw_obj):
        """将原始特征转换为数值特征向量"""
        raise NotImplementedError

    def feature_vector(self, bytez, pe_binary):
        """直接从样本本身计算特征向量。只有在结合使用这两个函数能显著提高速度的情况下，才可以采用不同的方法。"""
        return self.process_raw_features(self.raw_features(bytez, pe_binary))

'''
(1)字节直方图
'''
class ByteHistogram(FeatureType):
    """
    整个二进制文件的字节直方图
    """

    name = 'histogram'
    dim = 256  # 256 个字节值

    def raw_features(self, bytez, pe_binary):
        # 统计每个字节值的频率（0-255）
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        # 将计数归一化为频率分布
        counts = np.array(raw_obj, dtype=np.float32)
        return counts / (counts.sum() or 1)  # 避免除以零

'''
(2)2d字节/熵直方图
'''
class ByteEntropyHistogram(FeatureType):
    """
    基于 (Saxe and Berlin, 2015) 的二维字节/熵直方图。
    大致近似字节值和局部熵的联合概率。
    更多信息请参见 https://arxiv.org/pdf/1508.03096.pdf 中的第 2.1.1 节。
    """

    name = 'byteentropy'
    dim = 256  # 熵直方图的维度

    def __init__(self, window=2048, step=1024):
        self.window = window  # 窗口大小
        self.step = step  # 滑动步长

    def _entropy_bin_counts(self, block):
        # 粗粒度直方图，每 16 个字节一个 bin
        # 对输入的字节块进行右移 4 位操作，然后统计每个值出现的次数，得到 16 个 bin 的直方图
        c = np.bincount(block >> 4, minlength=16)
        # 计算每个 bin 的概率
        p = c.astype(np.float32) / self.window
        # 找出非零计数的 bin 的索引
        wh = np.where(c)[0]
        # 计算局部熵，乘以 2 是因为信息从 256 个 bin（8 位）减少到 16 个 bin（4 位）
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2
        # 将熵值映射到 0 - 15 的 bin 中
        Hbin = int(H * 2)
        # 处理熵值为 8.0 位的情况
        if Hbin == 16:
            Hbin = 15
        # 返回熵 bin 的索引和直方图计数
        return Hbin, c

    def raw_features(self, bytez, pe_binary):
        # 初始化一个 16x16 的零矩阵，用于存储字节/熵直方图
        output = np.zeros((16, 16), dtype=int)
        # 将字节数据转换为无符号 8 位整数数组
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            # 如果字节数据长度小于窗口大小，直接计算熵 bin 和直方图计数
            Hbin, c = self._entropy_bin_counts(a)
            # 将计数累加到对应的熵 bin 行
            output[Hbin, :] += c
        else:
            # 使用滑动窗口技巧生成块
            # 计算滑动窗口的形状
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            # 计算滑动窗口的步长
            strides = a.strides + (a.strides[-1],)
            # 使用 np.lib.stride_tricks.as_strided 生成滑动窗口块
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]
            # 遍历每个块，计算熵 bin 和直方图计数，并累加到输出矩阵中
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c
        # 将输出矩阵展平为一维列表并返回
        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        # 将原始特征转换为 float32 类型的 numpy 数组
        counts = np.array(raw_obj, dtype=np.float32)
        # 计算所有计数的总和
        total_sum = counts.sum()
        # 对计数进行归一化处理
        normalized = counts / total_sum
        # 返回归一化后的特征向量
        return normalized

'''
(3)节section信息
'''
class SectionInfo(FeatureType):
    """
    关于节名称、大小和熵的信息。使用哈希技巧
    将所有这些节信息总结为一个特征向量。
    """
    name = 'section'
    dim = 5 + 50 + 50 + 50 + 50 + 50
    def __init__(self):
        super(SectionInfo, self).__init__()
    @staticmethod
    def _properties(s):
        """
        获取节的特征属性列表。
        参数:
        s (pefile.SectionStructure): pefile 中的节结构对象。
        返回:
        list: 节的特征属性名称列表。
        """
        # pefile 中节的特征属性以整数形式存储，这里将其转换为字符串并提取关键部分
        characteristics = []
        # 节可执行属性
        if s.Characteristics & 0x20000000:
            characteristics.append('MEM_EXECUTE')
        # 节可读属性
        if s.Characteristics & 0x40000000:
            characteristics.append('MEM_READ')
        # 节可写属性
        if s.Characteristics & 0x80000000:
            characteristics.append('MEM_WRITE')
        return characteristics

    def raw_features(self, bytez, pe_binary):
        """
        提取文件的节相关原始特征。
        参数:
        bytez (bytes): 文件的字节数据。
        pe_binary (pefile.PE): 已解析的 PE 文件对象。
        返回:
        dict: 包含入口节名称和所有节详细信息的字典。
        """
        if pe_binary is None:
            return {"entry": "", "sections": []}
        entry_section = ""
        try:
            # 尝试获取入口点所在的节
            entry_rva = pe_binary.OPTIONAL_HEADER.AddressOfEntryPoint
            for section in pe_binary.sections:
                if section.VirtualAddress <= entry_rva < section.VirtualAddress + section.Misc_VirtualSize:
                    entry_section = section.Name.decode('utf-8').rstrip('\x00')
                    break
        except AttributeError:
            pass
        if not entry_section:
            # 若未找到入口点所在的节，查找第一个可执行节
            for section in pe_binary.sections:
                if section.Characteristics & 0x20000000:
                    entry_section = section.Name.decode('utf-8').rstrip('\x00')
                    break
        raw_obj = {
            "entry": entry_section,
            "sections": [
                {
                    'name': section.Name.decode('utf-8').rstrip('\x00'),
                    'size': section.SizeOfRawData,
                    'entropy': section.get_entropy(),
                    'vsize': section.Misc_VirtualSize,
                    'props': self._properties(section)
                } for section in pe_binary.sections
            ]
        }
        return raw_obj

    def process_raw_features(self, raw_obj):
        """
        处理原始特征，将其转换为特征向量。
        参数:
        raw_obj (dict): 包含入口节名称和所有节详细信息的字典。
        返回:
        np.ndarray: 处理后的特征向量。
        """
        sections = raw_obj['sections']
        general = [
            len(sections),  # 节的总数
            # 大小为零的节的数量
            sum(1 for s in sections if s['size'] == 0),
            # 名称为空的节的数量
            sum(1 for s in sections if s['name'] == ""),
            # 具有可读和可执行属性的节的数量
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            # 具有可写属性的节的数量
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        # 每个节的总体特征
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([[raw_obj['entry']]]).toarray()[0]
        characteristics = [p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]
        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
            characteristics_hashed
        ]).astype(np.float32)

'''
(4)导入表信息
'''
class ImportsInfo(FeatureType):
    """
    关于从导入地址表中导入的库和函数的信息。
    注意，导入函数的总数包含在 GeneralFileInfo中。
    """
    name = 'imports'
    dim = 1280
    def __init__(self):
        super(ImportsInfo, self).__init__()
    def raw_features(self, bytez, pe_binary):
        """
        提取文件的导入信息原始特征。
        参数:
        bytez (bytes): 文件的字节数据。
        pe_binary (pefile.PE): 已解析的 PE 文件对象。
        返回:
        dict: 包含导入库和对应导入函数列表的字典。
        """
        imports = {}
        if pe_binary is None:
            return imports
        try:
            # 遍历 PE 文件的导入表
            for entry in pe_binary.DIRECTORY_ENTRY_IMPORT:
                # 获取导入库的名称
                lib_name = entry.dll.decode('utf-8')
                if lib_name not in imports:
                    imports[lib_name] = []
                # 遍历每个库的导入函数
                for imp in entry.imports:
                    if imp.name is None:
                        # 如果是按序号导入
                        imports[lib_name].append("ordinal" + str(imp.ordinal))
                    else:
                        # 截取函数名前 10000 个字符
                        func_name = imp.name.decode('utf-8')[:10000]
                        imports[lib_name].append(func_name)
        except AttributeError:
            # 若文件没有导入表，直接返回空字典
            pass
        return imports
    def process_raw_features(self, raw_obj):
        """
        处理原始特征，将其转换为特征向量。
        参数:
        raw_obj (dict): 包含导入库和对应导入函数列表的字典。
        返回:
        np.ndarray: 处理后的特征向量。
        """
        # 获取唯一的导入库列表
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        # 使用 FeatureHasher 对库名进行哈希处理
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]
        # 生成每个导入函数的完整名称，格式为 "库名:函数名"
        imports = [lib.lower() + ':' + e for lib, elist in raw_obj.items() for e in elist]
        # 使用 FeatureHasher 对导入函数完整名称进行哈希处理
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]
        # 将库名哈希结果和导入函数哈希结果拼接成一个特征向量
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)


'''
(5)导出表信息
'''
class ExportsInfo(FeatureType):
    """
    关于导出函数的信息。注意，导出函数的总数包含在 GeneralFileInfo 中。
    """
    name = 'exports'
    dim = 128
    def __init__(self):
        super(ExportsInfo, self).__init__()
    def raw_features(self, bytez, pe_binary):
        """
        提取文件的导出函数原始特征。
        参数:
        bytez (bytes): 文件的字节数据。
        pe_binary (pefile.PE): 已解析的 PE 文件对象。
        返回:
        list: 包含导出函数名（截取前 10000 个字符）的列表。
        """
        if pe_binary is None:
            return []
        clipped_exports = []
        try:
            # 遍历 PE 文件的导出表
            for exp in pe_binary.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name is not None:
                    # 对导出函数名进行解码，并截取前 10000 个字符
                    func_name = exp.name.decode('utf-8')[:10000]
                    clipped_exports.append(func_name)
        except AttributeError:
            # 如果文件没有导出表，捕获 AttributeError 异常，直接返回空列表
            pass
        return clipped_exports
    def process_raw_features(self, raw_obj):
        """
        处理原始特征，将其转换为特征向量。
        参数:
        raw_obj (list): 包含导出函数名的列表。
        返回:
        np.ndarray: 处理后的特征向量。
        """
        # 使用 FeatureHasher 对导出函数名列表进行哈希处理，得到 128 维的哈希向量
        exports_hashed = FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0]
        return exports_hashed.astype(np.float32)

'''
(6)PE文件内部通用信息
'''
class GeneralFileInfo(FeatureType):
    """
    关于文件的通用信息
    """
    name = 'general'
    dim = 10
    def __init__(self):
        super(GeneralFileInfo, self).__init__()
    def raw_features(self, bytez, pe_binary):
        """
        提取文件的通用原始特征。
        参数:
        bytez (bytes): 文件的字节数据。
        pe_binary (pefile.PE): 已解析的 PE 文件对象。
        返回:
        dict: 包含文件通用信息的字典。
        """
        if pe_binary is None:
            return {
                'size': len(bytez),
                'vsize': 0,
                'has_debug': 0,
                'exports': 0,
                'imports': 0,
                'has_relocations': 0,
                'has_resources': 0,
                'has_signature': 0,
                'has_tls': 0,
                'symbols': 0
            }
        # 检查是否有调试信息
        has_debug = 1 if hasattr(pe_binary, 'DIRECTORY_ENTRY_DEBUG') and pe_binary.DIRECTORY_ENTRY_DEBUG else 0
        # 检查是否有导出表
        exports = len(pe_binary.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe_binary, 'DIRECTORY_ENTRY_EXPORT') else 0
        # 检查是否有导入表
        imports = len(pe_binary.DIRECTORY_ENTRY_IMPORT) if hasattr(pe_binary, 'DIRECTORY_ENTRY_IMPORT') else 0
        # 检查是否有重定位表
        has_relocations = 1 if hasattr(pe_binary, 'DIRECTORY_ENTRY_BASERELOC') and pe_binary.DIRECTORY_ENTRY_BASERELOC else 0
        # 检查是否有资源表
        has_resources = 1 if hasattr(pe_binary, 'DIRECTORY_ENTRY_RESOURCE') and pe_binary.DIRECTORY_ENTRY_RESOURCE else 0
        # pefile 没有直接判断签名的方法，这里暂时设为 0
        has_signature = 0
        # 检查是否有 TLS 表
        has_tls = 1 if hasattr(pe_binary, 'DIRECTORY_ENTRY_TLS') and pe_binary.DIRECTORY_ENTRY_TLS else 0
        # pefile 没有直接获取符号数量的方法，这里暂时设为 0
        symbols = 0
        return {
            'size': len(bytez),
            'vsize': pe_binary.OPTIONAL_HEADER.SizeOfImage,
            'has_debug': has_debug,
            'exports': exports,
            'imports': imports,
            'has_relocations': has_relocations,
            'has_resources': has_resources,
            'has_signature': has_signature,
            'has_tls': has_tls,
            'symbols': symbols
        }
    def process_raw_features(self, raw_obj):
        """
        处理原始特征，将其转换为特征向量。
        参数:
        raw_obj (dict): 包含文件通用信息的字典。
        返回:
        np.ndarray: 处理后的特征向量。
        """
        return np.asarray([
            raw_obj['size'], raw_obj['vsize'], raw_obj['has_debug'], raw_obj['exports'], raw_obj['imports'],
            raw_obj['has_relocations'], raw_obj['has_resources'], raw_obj['has_signature'], raw_obj['has_tls'],
            raw_obj['symbols']
        ], dtype=np.float32)

'''
(7)PE文件头信息
'''
class HeaderFileInfo(FeatureType):
    """
    从文件头中提取的机器、架构、操作系统、链接器等信息
    """
    name = 'header'
    dim = 62
    def __init__(self):
        super(HeaderFileInfo, self).__init__()
    def raw_features(self, bytez, pe_binary):
        # 初始化原始特征对象
        raw_obj = {'coff': {'timestamp': 0, 'machine': "", 'characteristics': []},
        'optional': {
            'subsystem': "",
            'dll_characteristics': [],
            'magic': "",
            'major_image_version': 0,
            'minor_image_version': 0,
            'major_linker_version': 0,
            'minor_linker_version': 0,
            'major_operating_system_version': 0,
            'minor_operating_system_version': 0,
            'major_subsystem_version': 0,
            'minor_subsystem_version': 0,
            'sizeof_code': 0,
            'sizeof_headers': 0,
            'sizeof_heap_commit': 0
        }
                   }
        if pe_binary is None:
            return raw_obj
        # 提取 COFF 头信息
        raw_obj['coff']['timestamp'] = pe_binary.FILE_HEADER.TimeDateStamp
        machine_mapping = {
            0x014c: 'I386',
            0x8664: 'AMD64'
            # 可根据需要添加更多机器类型映射
        }
        raw_obj['coff']['machine'] = machine_mapping.get(pe_binary.FILE_HEADER.Machine, "")
        characteristics_mapping = {
            0x0001: 'RELOCS_STRIPPED',
            0x0002: 'EXECUTABLE_IMAGE',
            0x0004: 'LINE_NUMS_STRIPPED',
            0x0008: 'LOCAL_SYMS_STRIPPED',
            0x0010: 'AGGRESIVE_WS_TRIM',
            0x0020: 'LARGE_ADDRESS_AWARE',
            0x0080: 'BYTES_REVERSED_LO',
            0x0100: '32BIT_MACHINE',
            0x0200: 'DEBUG_STRIPPED',
            0x0400: 'REMOVABLE_RUN_FROM_SWAP',
            0x0800: 'NET_RUN_FROM_SWAP',
            0x1000: 'SYSTEM',
            0x2000: 'DLL',
            0x4000: 'UP_SYSTEM_ONLY',
            0x8000: 'BYTES_REVERSED_HI'
        }
        # 使用位运算检查每个标志位
        characteristics = []
        for flag, name in characteristics_mapping.items():
            if pe_binary.FILE_HEADER.Characteristics & flag:
                characteristics.append(name)
        raw_obj['coff']['characteristics'] = characteristics
        # 提取可选头信息
        optional_header = pe_binary.OPTIONAL_HEADER
        subsystem_mapping = {
            1: 'NATIVE',
            2: 'WINDOWS_GUI',
            3: 'WINDOWS_CUI',
            7: 'POSIX_CUI',
            9: 'WINDOWS_CE_GUI',
            10: 'EFI_APPLICATION',
            11: 'EFI_BOOT_SERVICE_DRIVER',
            12: 'EFI_RUNTIME_DRIVER',
            13: 'EFI_ROM',
            14: 'XBOX',
            16: 'WINDOWS_BOOT_APPLICATION'
        }
        raw_obj['optional']['subsystem'] = subsystem_mapping.get(optional_header.Subsystem, "")
        dll_characteristics_mapping = {
            0x0040: 'DYNAMIC_BASE',
            0x0080: 'FORCE_INTEGRITY',
            0x0100: 'NX_COMPAT',
            0x0200: 'NO_ISOLATION',
            0x0400: 'NO_SEH',
            0x0800: 'NO_BIND',
            0x1000: 'APPCONTAINER',
            0x2000: 'WDM_DRIVER',
            0x4000: 'GUARD_CF',
            0x8000: 'TERMINAL_SERVER_AWARE'
        }
        # 同样使用位运算处理 DLL 特征标志
        dll_characteristics = []
        for flag, name in dll_characteristics_mapping.items():
            if optional_header.DllCharacteristics & flag:
                dll_characteristics.append(name)
        magic_mapping = {
            0x10b: 'PE32',
            0x20b: 'PE32+'
        }
        raw_obj['optional']['magic'] = magic_mapping.get(optional_header.Magic, "")
        raw_obj['optional']['major_image_version'] = optional_header.MajorImageVersion
        raw_obj['optional']['minor_image_version'] = optional_header.MinorImageVersion
        raw_obj['optional']['major_linker_version'] = optional_header.MajorLinkerVersion
        raw_obj['optional']['minor_linker_version'] = optional_header.MinorLinkerVersion
        raw_obj['optional']['major_operating_system_version'] = optional_header.MajorOperatingSystemVersion
        raw_obj['optional']['minor_operating_system_version'] = optional_header.MinorOperatingSystemVersion
        raw_obj['optional']['major_subsystem_version'] = optional_header.MajorSubsystemVersion
        raw_obj['optional']['minor_subsystem_version'] = optional_header.MinorSubsystemVersion
        raw_obj['optional']['sizeof_code'] = optional_header.SizeOfCode
        raw_obj['optional']['sizeof_headers'] = optional_header.SizeOfHeaders
        raw_obj['optional']['sizeof_heap_commit'] = optional_header.SizeOfHeapCommit
        return raw_obj

    def process_raw_features(self, raw_obj):
        # 处理原始特征，将其转换为特征向量
        return np.hstack([
            raw_obj['coff']['timestamp'],
            FeatureHasher(10, input_type="string").transform([[raw_obj['coff']['machine']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['coff']['characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['subsystem']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['optional']['dll_characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['magic']]]).toarray()[0],
            raw_obj['optional']['major_image_version'],
            raw_obj['optional']['minor_image_version'],
            raw_obj['optional']['major_linker_version'],
            raw_obj['optional']['minor_linker_version'],
            raw_obj['optional']['major_operating_system_version'],
            raw_obj['optional']['minor_operating_system_version'],
            raw_obj['optional']['major_subsystem_version'],
            raw_obj['optional']['minor_subsystem_version'],
            raw_obj['optional']['sizeof_code'],
            raw_obj['optional']['sizeof_headers'],
            raw_obj['optional']['sizeof_heap_commit'],
        ]).astype(np.float32)

'''
(8)字符串信息
'''
class StringExtractor(FeatureType):
    """
    从原始字节流中提取字符串
    """
    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1
    def __init__(self):
        super(StringExtractor, self).__init__()
        # 匹配连续 5 个及以上的可打印 ASCII 字符
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        # 匹配 'C:\' 字符串
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # 匹配 http:// 或 https:// 字符串
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # 匹配 'HKEY_' 前缀的字符串
        self._registry = re.compile(b'HKEY_')
        # 匹配 'MZ' 字符串
        self._mz = re.compile(b'MZ')
    def raw_features(self, bytez, pe_binary):
        # 提取所有匹配的字符串
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # 计算字符串长度统计信息
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # 将可打印字符映射到 0 - 95 的整数数组
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # 直方图计数
            # 计算可打印字符的分布
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # 熵
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0
        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),  # 存储非归一化的直方图
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }
    def process_raw_features(self, raw_obj):
        # 处理原始特征，将其转换为特征向量
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            np.asarray(raw_obj['printabledist']) / hist_divisor, raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)

'''
(9)数据目录
'''
class DataDirectories(FeatureType):
    """
    提取前15个数据目录的大小和虚拟地址
    """
    name = 'datadirectories'
    dim = 15 * 2
    def __init__(self):
        super(DataDirectories, self).__init__()
        self._name_order = [
            "EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
            "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE",
            "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"
        ]
    def raw_features(self, bytez, pe_binary):
        output = []
        if pe_binary is None:
            return output
        directory_mapping = {
            0: "EXPORT_TABLE",
            1: "IMPORT_TABLE",
            2: "RESOURCE_TABLE",
            3: "EXCEPTION_TABLE",
            4: "CERTIFICATE_TABLE",
            5: "BASE_RELOCATION_TABLE",
            6: "DEBUG",
            7: "ARCHITECTURE",
            8: "GLOBAL_PTR",
            9: "TLS_TABLE",
            10: "LOAD_CONFIG_TABLE",
            11: "BOUND_IMPORT",
            12: "IAT",
            13: "DELAY_IMPORT_DESCRIPTOR",
            14: "CLR_RUNTIME_HEADER"
        }
        for i, directory in enumerate(pe_binary.OPTIONAL_HEADER.DATA_DIRECTORY):
            if i < 15:
                output.append({
                    "name": directory_mapping[i],
                    "size": directory.Size,
                    "virtual_address": directory.VirtualAddress
                })
        return output
    def process_raw_features(self, raw_obj):
        # 处理原始特征，将其转换为特征向量
        features = np.zeros(2 * len(self._name_order), dtype=np.float32)
        for i in range(len(self._name_order)):
            if i < len(raw_obj):
                features[2 * i] = raw_obj[i]["size"]
                features[2 * i + 1] = raw_obj[i]["virtual_address"]
        return features


def extract_features_ember(samples_dir):
    """
    从样本目录提取特征。
    参数:
        samples_dir (list): [(文件路径, 标签), ...]
    返回:
        features (np.ndarray): 特征矩阵，形状为 (样本数, 特征维度)
        labels (np.ndarray): 标签数组，形状为 (样本数,)
    """
    features = []
    labels = []

    features_extractors = [
        ByteHistogram(),
        ByteEntropyHistogram(),
        SectionInfo(),
        ImportsInfo(),
        ExportsInfo(),
        GeneralFileInfo(),
        HeaderFileInfo(),
        StringExtractor(),
        DataDirectories()
    ]

    for sample_path, label in samples_dir:
        try:
            with open(sample_path, 'rb') as f:
                bytez = f.read()

            # 解析 PE 文件
            try:
                pe_binary = pefile.PE(data=bytez)  # 使用 pefile 解析
            except pefile.PEFormatError:
                print(f"无法解析 PE 文件: {sample_path}")
                continue
            # 提取所有特征
            feature_vectors = [fe.process_raw_features(fe.raw_features(bytez, pe_binary)) for fe in features_extractors]
            # 合并特征
            sample_features = np.concatenate(feature_vectors, axis=0)
            features.append(sample_features)
            labels.append(label)

        except Exception as e:
            print(f"处理样本 {sample_path} 时出错: {e}")
            continue

    return np.array(features), np.array(labels)


'''
malgraph图特征ACFG、FCG、CFG
使用json保存词表等数据
'''
'''
词频类
'''
class Vocab:
    def __init__(self, freq_file: str, max_vocab_size: int, min_freq: int = 1, unk_token: str = '<unk>',
                 pad_token: str = '<pad>', special_tokens: list = None):

        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens

        assert os.path.exists(freq_file), "The file of {} is not exist".format(freq_file)
        freq_counter = self.load_freq_counter_from_file(file_path=freq_file, min_freq=self.min_freq)

        self.token_2_index, self.index_2_token = self.create_vocabulary(freq_counter=freq_counter)

        self.unk_idx = None if self.unk_token is None else self.token_2_index[self.unk_token]
        self.pad_idx = None if self.pad_token is None else self.token_2_index[self.pad_token]

    def __len__(self):
        return len(self.index_2_token)

    def __getitem__(self, item: str):
        assert isinstance(item, str)
        if item in self.token_2_index.keys():
            return self.token_2_index[item]
        else:
            if self.unk_token is not None:
                return self.token_2_index[self.unk_token]
            else:
                raise KeyError("{} is not in the vocabulary, and self.unk_token is None".format(item))

    def create_vocabulary(self, freq_counter: Counter):

        token_2_index = {}  # dict
        index_2_token = []  # list

        if self.unk_token is not None:
            index_2_token.append(self.unk_token)
        if self.pad_token is not None:
            index_2_token.append(self.pad_token)
        if self.special_tokens is not None:
            for token in self.special_tokens:
                index_2_token.append(token)

        for f_name, count in tqdm(freq_counter.most_common(self.max_vocab_size), desc="creating vocab ... "):
            if f_name in index_2_token:
                print("trying to add {} to the vocabulary, but it already exists !!!".format(f_name))
                continue
            else:
                index_2_token.append(f_name)

        for index, token in enumerate(index_2_token):  # reverse
            token_2_index.update({token: index})

        return token_2_index, index_2_token

    @staticmethod
    def load_freq_counter_from_file(file_path: str, min_freq: int):
        freq_dict = {}
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Load frequency list from the file of {} ... ".format(file_path)):
                line = json.loads(line)
                f_name = line["f_name"]
                count = int(line["count"])

                assert f_name not in freq_dict, "trying to add {} to the vocabulary, but it already exists !!!"
                if count < min_freq:
                    print(line, "break")
                    break

                freq_dict[f_name] = count
        return Counter(freq_dict)

'''
(1)计算文件hash
'''
def calculate_file_hash(file_path):
    """
    计算文件的 SHA256 哈希值
    :param file_path: str 文件路径
    :return: str 文件的 SHA256 哈希值
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()
'''
(2) 基于Binary Ninja 提取CFG和FCG
'''
def extract_cfg_and_fcg(sample_path, vocab_dict):
    """
    提取函数调用图（FCG）和控制流图（CFG）
    :param sample_path: str PE 文件路径
    :param vocab_dict: dict
    :return: dict 包含 function_edges、acfg_list 等
    """
    try:
        # 计算文件哈希
        file_hash = calculate_file_hash(sample_path)
        # 打开 PE 文件
        bv = BinaryViewType["PE"].open(sample_path)
        if not bv:
            raise ValueError(f"无法打开文件：{sample_path}")
        bv.update_analysis_and_wait()

        for extern_func in bv.get_symbols_of_type(SymbolType.ImportAddressSymbol):
            if extern_func.name in vocab_dict:
                vocab_dict[extern_func.name] = vocab_dict[extern_func.name] + 1
            else:
                vocab_dict[extern_func.name] = 1

        # 提取本地函数和外部函数
        local_functions = list(bv.functions)
        external_functions = [
            sym.name for sym in bv.get_symbols_of_type(SymbolType.ImportAddressSymbol)
        ]

        local_function_count = len(local_functions)
        total_function_count = local_function_count + len(external_functions)

        # 构建 function_edges
        function_edges = [[], []]  # 第一个数组为调用者索引，第二个数组为被调用者索引

        # 构建函数索引映射
        function_index = {}
        for idx, func in enumerate(local_functions):
            function_index[func.start] = idx
        for idx, name in enumerate(external_functions):
            function_index[name] = idx + local_function_count
        # 构建导入地址映射
        external_symbols = bv.get_symbols_of_type(SymbolType.ImportAddressSymbol)
        import_address_map = {sym.address: sym.name for sym in external_symbols}

        for func in local_functions:
            caller_idx = local_functions.index(func)

            # 遍历 MLIL 指令，查找调用指令
            for block in func.mlil:
                for instr in block:
                    if instr.operation == MediumLevelILOperation.MLIL_CALL:
                        dest = instr.dest
                        callee_idx = None
                        if dest.operation == MediumLevelILOperation.MLIL_CONST_PTR:
                            # 直接调用目标
                            call_address = dest.constant
                            if call_address in function_index:
                                callee_idx = function_index[call_address]
                            elif call_address in import_address_map:
                                callee_name = import_address_map[call_address]
                                callee_idx = function_index[callee_name]
                        elif dest.operation == MediumLevelILOperation.MLIL_IMPORT:
                            # 导入函数调用
                            symbol = bv.get_symbol_at(instr.address)
                            if symbol and symbol.name in function_index:
                                callee_idx = function_index[symbol.name]
                        else:
                            # 处理其他情况，如间接调用
                            continue

                        if callee_idx is not None:
                            function_edges[0].append(caller_idx)
                            function_edges[1].append(callee_idx)

        # 提取 CFG 数据
        acfg_list = []
        for func in local_functions:
            cfg_data = {
                "block_number": len(func.basic_blocks),
                "block_edges": [[], []],  # 存储基本块的边信息
                "block_features": []  # 存储基本块特征
            }

            # 构建基本块映射
            block_map = {block: idx for idx, block in enumerate(func.basic_blocks)}

            # 提取 block_features 和 block_edges
            for block in func.basic_blocks:
                # 初始化特征统计
                feature_counts = {
                    "call": 0,
                    "transfer": 0,
                    "arithmetic": 0,
                    "logic": 0,
                    "compare": 0,
                    "move": 0,
                    "termination": 0,
                    "data_declaration": 0,
                    "total_instructions": 0,
                    "constants": 0,
                }

                # 遍历指令并统计特征
                for instr in block.disassembly_text:
                    if hasattr(instr, "tokens"):
                        feature_counts["total_instructions"] += 1
                        for token in instr.tokens:
                            if token.text in ["call"]:
                                feature_counts["call"] += 1
                            elif token.text in ["jmp", "jz", "jnz", "je", "jne"]:
                                feature_counts["transfer"] += 1
                            elif token.text in ["add", "sub", "mul", "div"]:
                                feature_counts["arithmetic"] += 1
                            elif token.text in ["and", "or", "xor"]:
                                feature_counts["logic"] += 1
                            elif token.text in ["cmp"]:
                                feature_counts["compare"] += 1
                            elif token.text in ["mov"]:
                                feature_counts["move"] += 1
                            elif token.text in ["ret", "hlt"]:
                                feature_counts["termination"] += 1
                            elif token.text in ["db", "dw", "dd"]:
                                feature_counts["data_declaration"] += 1
                            elif token.type.name in ["IntegerToken", "StringToken"]:
                                feature_counts["constants"] += 1

                # 后继基本块数量
                offspring_count = len(block.outgoing_edges)

                # 记录特征
                block_features = [
                    feature_counts["call"],
                    feature_counts["transfer"],
                    feature_counts["arithmetic"],
                    feature_counts["logic"],
                    feature_counts["compare"],
                    feature_counts["move"],
                    feature_counts["termination"],
                    feature_counts["data_declaration"],
                    feature_counts["total_instructions"],
                    feature_counts["constants"],
                    offspring_count,
                ]
                cfg_data["block_features"].append(block_features)

                # 提取基本块之间的边
                for edge in block.outgoing_edges:
                    src_idx = block_map.get(block)
                    dst_idx = block_map.get(edge.target)
                    if src_idx is not None and dst_idx is not None:
                        cfg_data["block_edges"][0].append(src_idx)
                        cfg_data["block_edges"][1].append(dst_idx)

            acfg_list.append(cfg_data)

        # 构建结果
        results = {
            "function_edges": function_edges,
            "acfg_list": acfg_list,
            "function_names": [func.name for func in local_functions] + external_functions,
            "function_number": total_function_count,
            "hash": file_hash,  # 文件的哈希值
        }
        return results

    except Exception as e:
        print(f"处理文件时出错：{e}")
        return None


def extract_features_malgraph(samples_dir):
    """
    包含function_edges、acfg_list 等
    先提取特征，将每个样本的特征转换成json形式，最后组合成jsonl;
    从jsonl中取出每一个样本的json，将每一个json变成单个的pt;

    返回词表vocabulary, pt文件所在目录pt_dir
    """
    vocab_dict = {} # 训练集的vocab统计
    jsonl_path = "malgraph_features.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for sample_path, label in samples_dir:
            sample_features = extract_cfg_and_fcg(sample_path, vocab_dict)
            if sample_features:
                data_to_write = sample_features.copy()
                data_to_write['label'] = label
                jsonl_file.write(json.dumps(data_to_write) + '\n')

        # 保存词汇表
        train_vocab_file = "train_external_function_name_vocab.jsonl"
        with open(train_vocab_file, 'w', encoding='utf-8') as vocab_file:
            for func_name, count in vocab_dict.items():
                json.dump({"f_name": func_name, "count": count}, vocab_file) # type: ignore
                vocab_file.write('\n')

    # 加载词汇表
    max_vocab_size = 10000
    vocabulary = Vocab(freq_file=train_vocab_file, max_vocab_size=max_vocab_size)
    # 将jsonl转换为单个的pt文件
    pt_dir = "pt_files"
    parse_json_list_2_pyg_object(jsonl_file=jsonl_path, vocab=vocabulary, output_dir=pt_dir)

    return vocabulary, pt_dir

'''
胶囊网络所需特征
'''
'''
(1)将PE文件转换成灰度图
'''
def convert_pe_to_grayscale_image(pe_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 生成保存图像的文件名
    file_name = os.path.basename(pe_path).split('.')[0] + '.png'
    save_path = os.path.join(save_dir, file_name)

    # 检查灰度图是否已经存在
    if os.path.exists(save_path):
        return save_path

    try:
        with open(pe_path, 'rb') as f:
            byte_data = f.read()
        # CapNet要求的图像大小为 224x224，根据实际情况调整
        img_size = 224
        num_pixels = img_size * img_size
        byte_array = np.frombuffer(byte_data, dtype=np.uint8)
        if len(byte_array) < num_pixels:
            byte_array = np.pad(byte_array, (0, num_pixels - len(byte_array)), 'constant')
        byte_array = byte_array[:num_pixels]
        img_array = byte_array.reshape((img_size, img_size))
        img = Image.fromarray(img_array).convert('L')
        img.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error converting {pe_path} to grayscale image: {e}")
        return None


def extract_features_rcnf(samples_dir):
    features = []
    labels = []

    # 定义图像预处理转换，增大尺寸
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 通常建议使用 299x299 尺寸
        transforms.ToTensor()
    ])

    for sample_path, label in samples_dir:
        if os.path.exists(sample_path):
            # 转换并保存灰度图，获取保存后的路径
            saved_img_path = convert_pe_to_grayscale_image(sample_path, './saved_gray_img')
            if saved_img_path:
                try:
                    # 读取保存的灰度图
                    img = Image.open(saved_img_path).convert('L')
                    img = transform(img)
                    # 将单通道灰度图复制为 3 通道
                    img = img.repeat(3, 1, 1)
                    features.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error reading or processing saved image {saved_img_path}: {e}")

    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels


def extract_features_transforms(samples_dir):
    """
    从包含 (文件路径, 标签) 元组列表中提取特征并构建数据集
    :param samples_dir: 包含 (文件路径, 标签) 的元组列表
    :return: 构建好的 TensorDataset
    """
    seq_len = 512 # 模型期望的序列长度
    all_features = []
    all_labels = []
    for file_path, label in samples_dir:
        try:
            with open(file_path, 'rb') as f:
                pe_bytes = f.read()
            byte_values = [b for b in pe_bytes]
            if len(byte_values) > seq_len:
                byte_values = byte_values[:seq_len]
            elif len(byte_values) < seq_len:
                padding = [0] * (seq_len - len(byte_values))
                byte_values.extend(padding)
            features = torch.tensor(byte_values, dtype=torch.long)
            all_features.append(features)
            all_labels.append(label)
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")

    if all_features and all_labels:
        all_features = torch.stack(all_features)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        dataset = TensorDataset(all_features, all_labels)
        return dataset
    else:
        print("No valid features and labels were extracted.")
        return None


def extract_features_1d_cnn(samples_dir, seq_len=512):
    """
    从包含 (文件路径, 标签) 元组列表中提取指令特征
    :param samples_dir: 包含 (文件路径, 标签) 的元组列表
    :param seq_len: 输入序列的长度
    :return: 特征数组和标签数组
    """
    features = []
    labels = []

    for sample_path, label in samples_dir:
        try:
            binary_view = BinaryViewType["PE"].open(sample_path)
            if binary_view is None:
                continue
            instructions = []
            for function in binary_view.functions:
                for block in function:
                    for instr in block:
                        instructions.append(instr[2])

            encoded_instructions = []
            for instruction in instructions:
                encoded = [ord(c) for c in instruction]
                encoded_instructions.extend(encoded)

            if len(encoded_instructions) > seq_len:
                encoded_instructions = encoded_instructions[:seq_len]
            elif len(encoded_instructions) < seq_len:
                padding = [0] * (seq_len - len(encoded_instructions))
                encoded_instructions.extend(padding)

            features.append(encoded_instructions)
            labels.append(label)
        except Exception as e:
            print(f"Error extracting features from {sample_path}: {e}")

    if features and labels:
        features = np.array(features)
        labels = np.array(labels)
        return features, labels
    else:
        print("No valid features and labels were extracted.")
        return None, None


# 3. 将特征和标签保存为HDF5文件/json文件形式
def save_features_to_hdf5(features, labels, h5_file):
    """
    将特征和标签存储到 HDF5 文件中。
    参数:
        features (Tensor): 特征张量 (batch_size, max_len)
        labels (Tensor): 标签张量 (batch_size,)
        h5_file (str): 存储的 HDF5 文件路径
    """
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('features', data=features.numpy())
        f.create_dataset('labels', data=labels.numpy())

def parse_json_list_2_pyg_object(jsonl_file: str, vocab: Vocab, output_dir: str):
    index = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with open(jsonl_file, "r", encoding="utf-8") as file:
            for item in tqdm(file):
                try:
                    item = json.loads(item)
                    item_hash = item['hash']
                    label = item.pop('label')
                    acfg_list = []
                    for one_acfg in item['acfg_list']:  # list of dict of acfg
                        block_features = one_acfg['block_features']
                        block_edges = one_acfg['block_edges']
                        one_acfg_data = Data(x=torch.tensor(block_features, dtype=torch.float),
                                             edge_index=torch.tensor(block_edges, dtype=torch.long))
                        acfg_list.append(one_acfg_data)
                    item_function_names = item['function_names']
                    item_function_edges = item['function_edges']
                    local_function_name_list = item_function_names[:len(acfg_list)]
                    assert len(acfg_list) == len(
                        local_function_name_list), "The length of ACFG_List should be equal to the length of Local_Function_List"
                    external_function_name_list = item_function_names[len(acfg_list):]
                    external_function_index_list = [vocab[f_name] for f_name in external_function_name_list]
                    index += 1
                    output_path = os.path.join(output_dir, "{}.pt".format(index))
                    torch.save(
                        Data(hash=item_hash, local_acfgs=acfg_list, external_list=external_function_index_list,
                             function_edges=item_function_edges, targets=label), output_path)
                except Exception as e:
                    print(f"Error processing item {index}: {e}")
    except Exception as e:
        print(f"Error reading {jsonl_file}: {e}")
