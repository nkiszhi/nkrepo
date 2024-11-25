import os, math
import numpy as np

def get_all_file_path(path) -> list:
    result = []
    for file in os.listdir(path):
        p = os.path.join(path, file)
        if os.path.isdir(p):
            result += get_all_file_path(p)
        else:
            result.append(p)
    return result

def bytes2int8(data : bytes) -> np.array:
    # 从字节数据创建一个numpy数组，数据类型为uint8
    return np.frombuffer(data, dtype=np.uint8)

def size_norm(data, length, fill = b"\x00") :
    # 删除多余的字节或填充data，使data的长度为length
    if len(data) >= length :
        return data[:length]
    else :
        return data + fill * (length - len(data))

def bytes_to_img_gray(data, fill=b"\x00"):
    # ceil向上取整，sqrt取平方根
    size = math.ceil(math.sqrt(len(data)))
    data = size_norm(data, size * size, fill=fill)
    data = bytes2int8(data)
    data = data.reshape((size, size, 1))
    # 将data重构成size*size的矩阵，1通道（灰度图像）
    return data

def bytes_to_img_col(data, fill=b"\x00"):
    size = math.ceil(math.sqrt(len(data) / 3))
    data = size_norm(data, size * size * 3, fill=fill)
    data = bytes2int8(data)
    data = data.reshape((size, size, 3))
    # 3通道图片
    return data

def bytes_to_seq(data):
    data = bytes2int8(data)
    data = data.reshape((len(data), 1, 1))
    return data