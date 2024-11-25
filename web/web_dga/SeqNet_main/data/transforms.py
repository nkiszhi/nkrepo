import cv2
import torch
import torchvision.transforms as transforms
from SeqNet_main.utils.utils import *

trans_malnet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=0.5, std=0.5),
])

trans_vision = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=0.5, std=0.5),
])

trans_seq = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((2**18, 1)), # 2**18
    transforms.Normalize(mean=0.5, std=0.5),
])

trans_seq_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5),
])

def malnet_transform(data):
    data = bytes_to_img_gray(data)
    data = trans_malnet(data.copy())
    return data

def vision_transform(data):
    data = bytes_to_img_col(data)
    data = trans_vision(data.copy())
    return data

def malconv_transform(data):
    data = size_norm(data, 2000000, fill = b"\x00") # 默认就是b"\x00"
    data = bytes2int8(data)
    return torch.tensor(data, dtype=torch.int32)

def malconv_gct_transform(data):
    data = size_norm(data, 16000000)
    data = bytes2int8(data)
    return torch.tensor(data, dtype=torch.int32)

def seq_transform(data):
    data = bytes_to_seq(data)
    data = trans_seq(data.copy())
    #如果第2个维度的长度为1 将第2个维度去掉
    return data.squeeze(2)

def seq_transform_norm(data):
    data = bytes_to_seq(data)
    # 使用线性插值将data resize到1，2**18
    # assert not data is None
    try:
        res = cv2.resize(data, (1, 2 ** 18), interpolation=cv2.INTER_LINEAR)
        data = trans_seq_norm(res)
    except:
        for i in data:
            if i < 0:
                print(i)
        assert 0
    
    return data.squeeze(2)

def get_trans(net):
    if "regionnet" in net:
        return malnet_transform
    elif net == "resnet" or net == "mobilenet" or net == "mobilenetlarge" or net == "mobilenetsmall":
        return vision_transform
    elif net == "malconv":
        return malconv_transform
    elif net == "malconv2":
        return malconv_gct_transform
    elif net == "seqnetnorm":
        return seq_transform_norm
    elif "seqnet" in net:
        return seq_transform
    else:
        raise Exception("No such model:", net)