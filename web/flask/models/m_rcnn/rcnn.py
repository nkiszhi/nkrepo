import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from configparser import ConfigParser
from models.m_rcnn.extract_feature import scan_load_samples, scan_load_prediction_samples, extract_features_rcnn

# Load configuration
cp = ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini'))
TRAINING_DATA = cp.get('files', 'training_data')
MODEL_PATH = cp.get('files', 'model_path')


class RCNN(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_channels,
        window_size,
        module,
        hidden_size,
        num_layers,
        bidirectional,
        residual,
        dropout=0.5,
    ):
        super(RCNN, self).__init__()
        assert module.__name__ in {"RNN", "GRU", "LSTM"}, "`module` 必须是 PyTorch 循环层"
        self.residual = residual
        self.embed = nn.Embedding(257, embed_dim)  # 索引范围 [0, 256]
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=window_size,
            stride=1,
            padding=1
        )
        self.rnn = module(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        rnn_out_size = (int(bidirectional) + 1) * hidden_size
        if residual:
            self.fc = nn.Linear(out_channels + rnn_out_size, 1)
        else:
            self.fc = nn.Linear(rnn_out_size, 1)

    def forward(self, x):
        # x 形状: (batch_size, seq_length)
        embedding = self.dropout(self.embed(x))
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        if self.residual:
            values, _ = conv_out.max(dim=-1)
        conv_out = conv_out.permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv_out)
        fc_in = rnn_out[-1]
        if self.residual:
            fc_in = torch.cat((fc_in, values), dim=-1)
        output = self.fc(fc_in).squeeze(1)
        return output


def running_training():
    SAMPLES_DIR = TRAINING_DATA
    EPOCHS = 10
    LR = 0.001
    BATCH_SIZE = 32
    MODEL_PATH = "rcnn_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = extract_features_rcnn(scan_load_samples(SAMPLES_DIR))
    if features is None or labels is None:
        print("No valid features or labels extracted.")
        return

    features = features.clamp(0, 256).long()

    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型初始化
    model = RCNN(
        embed_dim=128,
        out_channels=64,
        window_size=3,
        module=nn.LSTM,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
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
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


#def running_predict(file_path):
#    predict_SAMPLES_DIR = file_path
#    MODEL_PATH = "rcnn_model.pth"
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#    model = RCNN(
#        embed_dim=128,
#        out_channels=64,
#        window_size=3,
#        module=nn.LSTM,
#        hidden_size=128,
#        num_layers=2,
#        bidirectional=True,
#        residual=True
#    ).to(device)
#    try:
#        model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device ('cpu'),weights_only=True))
#    except FileNotFoundError:
#        print(f"Model file {MODEL_PATH} not found. Please train the model first.")
#        return
#    model.eval()
#
#    features, _ = extract_features_rcnn(scan_load_prediction_samples(predict_SAMPLES_DIR))
#    if features is None:
#        print("No valid features extracted.")
#        return
#
#    features = features.clamp(0, 256).long().to(device)
#
#    with torch.no_grad():
#        predictions = model(features)
#        binary_predictions = (torch.sigmoid(predictions) > 0.5).float()
#
#    print("Prediction results:")
#    for i, pred in enumerate(binary_predictions):
#        print(f"Sample {i + 1}: {'Malicious' if pred.item() == 1 else 'Benign'}")
#
#    return binary_predictions

def run_prediction(file_path):
    predict_SAMPLES_DIR = file_path
    model_file_path = os.path.join(MODEL_PATH, 'm_rcnn', 'saved', 'rcnn_model.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RCNN(
        embed_dim=128,
        out_channels=64,
        window_size=3,
        module=nn.LSTM,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        residual=True
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu'), weights_only=True))
    except FileNotFoundError:
        print(f"Model file {model_file_path} not found. Please train the model first.")
        return None

    model.eval()

    features, _ = extract_features_rcnn(scan_load_prediction_samples(predict_SAMPLES_DIR))
    if features is None:
        print("No valid features extracted.")
        return None

    features = features.clamp(0, 256).long().to(device)

    with torch.no_grad():
        predictions = model(features)
        probabilities = torch.sigmoid(predictions)

    # 假设只处理一个样本，返回该样本的预测概率
    if probabilities.size(0) > 0:
        score = probabilities[0].item()
    else:
        score = None

    return score


if __name__ == '__main__':
    # running_training()
    running_predict()