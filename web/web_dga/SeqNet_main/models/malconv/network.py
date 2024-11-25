import torch
import torch.nn as nn
import torch.nn.functional as F

class MalConv(nn.Module):
    def __init__(self, input_size : int, embedding_size = 257, kern_size = 512, stride_size = 512):
        super(MalConv, self).__init__()
        self.input_size = input_size

        # embedding_size个tensors，每个的size是8（257*8）
        # embedding_size = 257 对文件的每个byte进行转换
        # 8维bedding
        self.embedding = nn.Embedding(embedding_size, 8) 
        
        # nn.Conv1d(in_channels, out_channels, ……) 
        # in_channels, 输入信号的通道，在1维卷积中，指的是词向量的维度
        # out_channels, 有多少个1维卷积

        # 将8维embedding的结果分成两个四维A和B
        self.conv1 = nn.Conv1d(4, 128, kernel_size=kern_size, stride=stride_size)
        self.conv2 = nn.Conv1d(4, 128, kernel_size=kern_size, stride=stride_size)


        self.maxpool = nn.MaxPool1d(1)
        self.flatten = nn.Flatten()
        # print((embedding_size - kern_size) // kern_size + 1)

        # 128维的全连接层
        self.fulc1 = nn.Linear(128 * ((input_size - kern_size) // kern_size + 1), 128)

        # 将前一层的128个特征映射到2个输出特征
        self.fulc2 = nn.Linear(128, 2)

    def forward(self, x):
        # if x.shape[1] != self.input_size :
        #     raise Exception("The input size should be {}, but input {}".format(self.input_size, x.shape[1]))
        
        # x 3个维度，batch_size, text_len, embedding_size
        x = self.embedding(x)
        # 交换矩阵的两个维度
        # 将x转换成 batch_size, embedding_size, text_len
        x = torch.transpose(x, 1, 2)

        # 分别对前四个维度和后四个维度进行卷积
        conv1_result = self.conv1(x[:, :4, :])
        conv2_result = F.sigmoid(self.conv2(x[:, 4:, :]))

        x = conv2_result * conv1_result
        # x = F.relu(x)

        # 最大池化
        x = self.maxpool(x)
        # 将维度转为一维
        x = self.flatten(x)

        # x = x.view(-1, 128)
        x = F.relu(self.fulc1(x))
        x = F.softmax(self.fulc2(x))
        return x

    def attack_forward(self, x):
        x = torch.transpose(x, 1, 2)
        conv1_result = self.conv1(x[:, :4, :])
        conv2_result = F.sigmoid(self.conv2(x[:, 4:, :]))

        x = conv2_result * conv1_result
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        # #
        x = F.relu(self.fulc1(x))
        x = self.fulc2(x)
        return x