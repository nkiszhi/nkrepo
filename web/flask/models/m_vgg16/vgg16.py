import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader

from models.m_vgg16.extract_feature import scan_load_samples, scan_load_prediction_samples

# 全局变量
TRAINING_SAMPLE_DIR = r"E:\Experimental data\dr_data"
MODEL_PATH = "/home/nkamg/nkrepo/zjp/multi_model_detection_system/new_flask/models/m_vgg16/saved/trained_vgg16_model.pth"


def pe_to_image(file_path, image_size=(224, 224)):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        byte_array = np.frombuffer(data, dtype=np.uint8)
        side_length = int(np.ceil(np.sqrt(len(byte_array))))
        padded_array = np.pad(byte_array, (0, side_length * side_length - len(byte_array)), mode='constant')
        image_array = padded_array.reshape((side_length, side_length))
        image = Image.fromarray(image_array, mode='L')
        image = image.resize(image_size)
        return image
    except Exception as e:
        print(f"将 {file_path} 转换为图像时出错: {e}")
        return None


def extract_features_vgg16(samples):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_images = []
    all_labels = []
    for file_path, label in samples:
        image = pe_to_image(file_path)
        if image is not None:
            input_tensor = transform(image).unsqueeze(0)
            all_images.append(input_tensor)
            all_labels.append(label)

    if all_images:
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
    return all_images, all_labels


def running_train():
    train_samples = scan_load_samples(TRAINING_SAMPLE_DIR)
    images, labels = extract_features_vgg16(train_samples)

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    weights = models.VGG16_Weights.IMAGENET1K_V1
    vgg16 = models.vgg16(weights=weights)
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg16.to(device)

    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = vgg16(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    print('Training finished.')
    torch.save(vgg16.state_dict(), MODEL_PATH)
    return


def run_prediction(file_path):
    try:
        # 确保输入格式为 [(file_path, -1)] 形式，若不是则进行转换
        if isinstance(file_path, str):
            predict_samples = [(file_path, -1)]
        else:
            predict_samples = file_path

        images, _ = extract_features_vgg16(predict_samples)

        if images is None or images.numel() == 0:
            print("未提取到有效的图像特征。")
            return None

        vgg16 = models.vgg16(weights=None)
        num_ftrs = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_ftrs, 2)

        vgg16.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
        vgg16.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg16.to(device)

        all_probabilities = []
        file_paths = [sample[0] for sample in predict_samples]
        idx_to_label = {0: 'benign', 1: 'malware'}

        with torch.no_grad():
            for inputs in DataLoader(images, batch_size=32, shuffle=False):
                inputs = inputs.to(device)
                outputs = vgg16(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 获取预测为恶意类别的概率
                all_probabilities.extend(probabilities.cpu().tolist())

        # 假设只处理一个样本，返回该样本的预测概率
        if all_probabilities:
            score = all_probabilities[0]
        else:
            score = None

        print("预测结果：")
        for file_path, prob in zip(file_paths, all_probabilities):
            label = idx_to_label[1] if prob > 0.5 else idx_to_label[0]
            print(f"文件: {file_path}, 预测类别: {label}, 预测概率: {prob:.4f}")

        return score
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        return None


if __name__ == "__main__":
    # running_train()
    file_path = "/your/test/file/path"  # 替换为实际的测试文件路径
    score = run_prediction(file_path)
    if score is not None:
        result = '恶意' if score > 0.5 else '安全'
        print(f"最终预测结果: 概率 {score:.4f}, 判定 {result}")


