import argparse, os

import torch, gc

from models.model import MalNet
from data.transforms import get_trans
from data.maldataset import MalDataModule
from pytorch_lightning.trainer import Trainer
from utils.test_val import test_model, val_model
from pytorch_lightning.callbacks import ModelCheckpoint
from models.seqnet.network import SequenceNet
from models.seqnet.seqops import *

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("option", help="train / test / val")
    parser.add_argument("--model", default="seqnet", help="malnet / malconv / resnet / mobilenet / seqnet ")
    parser.add_argument("--logdir", default="./log", help="path for logger")
    parser.add_argument("--val_path", default="./data/NewDataset/val/", help="path for validation")
    parser.add_argument("--train_path", default="./data/NewDataset/train/", help="path for training")
    parser.add_argument("--checkpoint", default=None, help="checkpoint for continue training, testing and validating")
    parser.add_argument("--test_file", default=None, help="file path for testing")
    parser.add_argument("--batch_size", default=32, help="Batch size for training")
    parser.add_argument("--state", default="start", help="continue or start, default: start")
    parser.add_argument("--mal", default="mal", help="Malware label for dataset")
    parser.add_argument("--norm", default="norm", help="Benign label for dataset")
    parser.add_argument("--epoch", default="70", help="Max epoch")
    parser.add_argument("--gpus", default=1, help="GPUs for training")

    return vars(parser.parse_args())

def main(args):
    seed_everything()
    gc.collect()
    torch.cuda.empty_cache()
    logdir = os.path.join(args["logdir"], args["model"])
    if args["option"] == "train":
        if args["state"] == "continue":
            if args["checkpoint"] is not None:
                model = MalNet.load_from_checkpoint(checkpoint_path=args["checkpoint"], logdir=logdir, net=args["model"])
                print("Continue training. Use checkpoint:", args["checkpoint"])
            else:
                raise Exception("When continue training, you need to assign a checkpoint path. Usage --checkpoint [path]")
        else:
            model = MalNet(logdir=logdir, net=args["model"])

        data = MalDataModule(
            mal_path=os.path.join(args["train_path"], args["mal"]), # 恶意样本路径
            norm_path=os.path.join(args["train_path"], args["norm"]), # 良性样本路径
            trans=get_trans(args["model"]), # 神经网络模型
            batch_size=int(args["batch_size"]), # 默认32
            val_mal=os.path.join(args["val_path"], args["mal"]), # 恶意样本的验证路径
            val_norm=os.path.join(args["val_path"], args["norm"]) # 良性样本的验证路径
        )
        # 保存准确率最高的3个权重模型
        checkpoint_callback = ModelCheckpoint(
            monitor="Accuracy", # 监控指标是准确率
            dirpath=os.path.join(logdir, "checkpoint"),
            filename=args["model"] + "-{epoch:02d}-{Accuracy:.4f}", # epoch轮次
            save_top_k=3, 
            mode="max",
        )
        trainer = Trainer(
            max_epochs=int(args["epoch"]), # 默认是70
            callbacks=[checkpoint_callback], # 回调函数 保存模型权重
            gpus=args["gpus"],
            default_root_dir=os.path.join(logdir, "tensorboard"), # tensorboard是用于可视化训练过程的工具，设置它的默认日志目录
        )
        # 使用上面创建的trainer对象开始训练模型
        trainer.fit(model, data)

    elif args["option"] == "test":
        if args["checkpoint"] is not None:
            model = MalNet.load_from_checkpoint(checkpoint_path=args["checkpoint"], logdir=logdir, net=args["model"])
            print("Testing. Use checkpoint:", args["checkpoint"])
        else:
            raise Exception("When testing, you need to assign a checkpoint path. Usage --checkpoint [path]")

        result = test_model(model, args["test_file"], get_trans(args["model"]))
        print("File:", args["test_file"])
        print("Benign:", result[0])
        print("Malicious:", result[1])

    elif args["option"] == "val":
        if args["checkpoint"] is not None:
            if args["checkpoint"].split(".")[-1] == "pkl":
                model = MalNet(logdir=logdir, net=args["model"])
                model.model.load_state_dict(torch.load(args["checkpoint"]).state_dict())
            else:
                model = MalNet.load_from_checkpoint(checkpoint_path=args["checkpoint"], logdir=logdir, net=args["model"])
            print("Validating. Use checkpoint:", args["checkpoint"])
        else:
            raise Exception("When validating, you need to assign a checkpoint path. Usage --checkpoint [path]")
        result, true_dis, false_dis, all_dis = val_model(model, val_mal=os.path.join(args["val_path"], args["mal"]), val_norm=os.path.join(args["val_path"], args["norm"]), trans=get_trans(args["model"]), batch_size=int(args["batch_size"]))
        print("Summary:")
        for key, value in result.items():
            print("\t", key, ":", value)
        print("\nDetail:")
        print("Correct distribution:")
        for index, dis in enumerate(true_dis):
            index = index / 10.
            # %.1f 保留一位小数的浮点数
            print("\t", "%.1f" % index, '~', "%.1f" % (index + 0.1), ':', dis)

        print("Wrong distribution:")
        for index, dis in enumerate(false_dis):
            index = index / 10.
            print("\t", "%.1f" % index, '~', "%.1f" % (index + 0.1), ':', dis)

        print("All distribution:")
        for index, dis in enumerate(all_dis):
            index = index / 10.
            print("\t", "%.1f" % index, '~', "%.1f" % (index + 0.1), ':', dis)     

    else:
        raise Exception("No option called", args["option"])


if __name__ == '__main__':
    main(get_args())
