import torch, os
from SeqNet_main.data.maldataset import MalwareDataset
from SeqNet_main.utils.utils import get_all_file_path
from torch.utils.data.dataloader import DataLoader
import random
import numpy as np

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def test_model(model, test_file, trans):
    # model = model.cuda()
    seed_everything()
    model.eval()
    with open(test_file, "rb") as f:
        # data = trans(f.read()).cuda()
        data = trans(f.read())
        data = data.unsqueeze(0)
        with torch.no_grad():
            result = model(data)
            # result = result.detach().cpu().numpy()[0] 
            result = result.detach().numpy()[0] 
            # detach表示不计算该tensor的梯度，只关心模型的输出
            # cpu(): 将Tensor从GPU移动到CPU。
            # numpy(): 将Tensor转换为Numpy数组。
            # [0]获取模型输出的概率或得分
            return result

def val_model(model, val_mal, val_norm, trans, batch_size):
    model = model.cuda() # 将模型加载到GPU上
    model.eval() # 不启动BatchNormalization和Dropout。预测的时候使用eval，防止出现不一致的预测结果
    if os.path.exists(val_mal):
        val_mal = get_all_file_path(val_mal)
    else:
        val_mal = []

    if os.path.exists(val_norm):
        val_norm = get_all_file_path(val_norm)
    else:
        val_norm = []

    dataset = MalwareDataset(mal_paths=val_mal, norm_paths=val_norm, trans=trans)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    print("Number:", len(dataloader), '*', batch_size)
    outputs = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.cuda()
            y = y.cuda()
            x = model(x) # 使用模型对数据进行预测
            outputs.append((x, y, model.get_loss(x, y))) # 将预测结果、真实标签和损失添加到输出列表中
        return output_process(outputs) # 调用output_process函数处理输出列表。计算准确率、精确率、召回率等指标

def output_process(outputs):
    # 四个用于计数的变量
    mal_t = 0 
    mal_f = 0
    norm_t = 0
    norm_f = 0
    all = 0
    loss_all = 0

    # 用于统计正确和错误预测的样本在每个分类的分布情况
    true_distribution = [0] * 10
    all_distribution = [0] * 10
    false_distribution = [0] * 10
    for output in outputs:
        all += len(output[1])
        loss_all += output[2].detach().cpu().numpy()
        for recogn, label in zip(output[0], output[1]):
            predict = torch.argmax(recogn)
            mal_possibility = int(recogn[1].detach().cpu().numpy() * 10)
            if mal_possibility >= 10:
                mal_possibility = 9
            elif mal_possibility < 0:
                mal_possibility = 0
            all_distribution[mal_possibility] += 1
            if predict == 1 and label == 1:
                mal_t += 1
                true_distribution[mal_possibility] += 1
            elif predict == 0 and label == 1:
                mal_f += 1
                false_distribution[mal_possibility] += 1
            elif predict == 1 and label == 0:
                norm_f += 1
                false_distribution[mal_possibility] += 1
            elif predict == 0 and label == 0:
                norm_t += 1
                true_distribution[mal_possibility] += 1

    # 确保所有样本都被正确地计数
    assert norm_f + norm_t + mal_f + mal_t == all
    assert sum(true_distribution) == mal_t + norm_t
    assert sum(false_distribution) == mal_f + norm_f
    assert sum(all_distribution) == all
    result = {
        "valLoss": loss_all / len(outputs),
        "Accuracy": (norm_t + mal_t) / all,
        "Wrong": (norm_f + mal_f) / all,
        "Precision": (mal_t / (mal_t + norm_f) if mal_t + norm_f > 0 else -1),
        "Recall": (mal_t / (mal_t + mal_f) if mal_t + mal_f > 0 else -1),
        "norm_f": norm_f,
        "norm_t": norm_t,
        "mal_f": mal_f,
        "mal_t": mal_t,
    }
    return result, true_distribution, false_distribution, all_distribution