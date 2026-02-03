import logging
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import pefile
from configparser import ConfigParser
from models.m_attention_rcnn.extract_feature import scan_load_samples, scan_load_prediction_samples, extract_features_attention_rcnn

# Load configuration
cp = ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini'))
TRAINING_DATA = cp.get('files', 'training_data')
MODEL_PATH = cp.get('files', 'model_path')


class AttentionRCNN(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_channels,
        window_size,
        module,
        hidden_size,
        num_layers,
        bidirectional,
        attn_size,
        residual,
        dropout=0.5,
    ):
        super(AttentionRCNN, self).__init__()
        assert module.__name__ in {"RNN", "GRU", "LSTM"}, "`module` 必须是 PyTorch 循环层"
        self.residual = residual
        self.embed = nn.Embedding(257, embed_dim)  # 注意：嵌入层输入需为整数索引（0-256）
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=window_size,
            stride=window_size,
        )
        self.rnn = module(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        rnn_out_size = (int(bidirectional) + 1) * hidden_size
        self.local2attn = nn.Linear(rnn_out_size, attn_size)
        self.global2attn = nn.Linear(rnn_out_size, attn_size, bias=False)
        self.attn_scale = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(attn_size, 1))
        )
        self.dropout = nn.Dropout(dropout)
        if residual:
            self.fc = nn.Linear(out_channels + rnn_out_size, 1)
        else:
            self.fc = nn.Linear(rnn_out_size, 1)

    def forward(self, x):
        # x 应为长整型张量（嵌入层要求）
        embedding = self.dropout(self.embed(x))  # 输入x必须是整数索引，范围[0, 256]
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        if self.residual:
            values, _ = conv_out.max(dim=-1)
        conv_out = conv_out.permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv_out)
        global_rnn_out = rnn_out.mean(dim=0)
        attention = torch.tanh(
            self.local2attn(rnn_out) + self.global2attn(global_rnn_out)
        ).permute(1, 0, 2)
        alpha = F.softmax(attention.matmul(self.attn_scale), dim=-1)
        rnn_out = rnn_out.permute(1, 0, 2)
        fc_in = (alpha * rnn_out).sum(dim=1)
        if self.residual:
            fc_in = torch.cat((fc_in, values), dim=-1)
        output = self.fc(fc_in).squeeze(1)
        return output


def running_training():
    base_dir = TRAINING_DATA
    EPOCHS = 10
    LR = 0.001
    BATCH_SIZE = 32
    MODEL_PATH = "attention_rcnn_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = extract_features_attention_rcnn(scan_load_samples(base_dir))

    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AttentionRCNN(
        embed_dim=128,
        out_channels=64,
        window_size=3,
        module=nn.LSTM,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        attn_size=64,
        residual=True
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, 损失: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型保存到 {MODEL_PATH}")
    return model

def run_prediction(file_path):
    sample_folder = file_path
    model_file_path = os.path.join(MODEL_PATH, 'm_attention_rcnn', 'saved', 'attention_rcnn_model.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionRCNN(
        embed_dim=128,
        out_channels=64,
        window_size=3,
        module=nn.LSTM,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        attn_size=64,
        residual=True
    ).to(device)
    
    try:
        # 加载模型（添加 map_location 避免设备错误）
        checkpoint = torch.load(
            model_file_path,
            map_location=torch.device('cpu'),  # 确保加载到 CPU（兼容无 GPU 环境）
            weights_only=True
        )
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print("错误：模型未找到")
        return None  # 返回 None 表示预测失败

    model.eval()
    samples = scan_load_prediction_samples(sample_folder)
    features, _ = extract_features_attention_rcnn(samples)  # 提取特征（忽略标签）

    if features is None or len(features) == 0:
        print("错误：未提取到有效特征")
        return None

    features = features.to(device)  # 移动到目标设备（CPU/GPU）

    with torch.no_grad():
        predictions = model(features)
        probabilities = torch.sigmoid(predictions)  # 获取概率值（0-1 之间）

    # 返回第一个样本的概率（假设输入为单样本，适配 3.py 的单文件预测场景）
    return probabilities[0].item() if len(probabilities) > 0 else None

if __name__ == '__main__':
    #running_training()
    running_predict()


