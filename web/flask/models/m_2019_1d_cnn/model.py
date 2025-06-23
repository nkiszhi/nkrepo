import torch
import torch.nn as nn
import torch.optim as optim


# 定义 1D - CNN 模型
class OneD_CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(OneD_CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        for _ in range(6):
            self.conv_layers.append(nn.Conv1d(input_channels, 16, kernel_size=3, stride=1))
            self.conv_layers.append(nn.ReLU())
            input_channels = 16
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16, 16)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.mean(dim=2)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


