import argparse, os

import torch
import sys

from SeqNet_main.models.model import MalNet
from SeqNet_main.data.transforms import get_trans
from SeqNet_main.data.maldataset import MalDataModule
from pytorch_lightning.trainer import Trainer
from SeqNet_main.utils.test_val import test_model, val_model
from pytorch_lightning.callbacks import ModelCheckpoint
from SeqNet_main.models.seqnet.network import SequenceNet
from SeqNet_main.models.seqnet.seqops import *

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

logdir="./SeqNet_main/log"
#model_list=["seqnet","malnet","malconv","resnet","mobilenet"]
model_list=["seqnet","malconv","mobilenet"]

def EXEDetection(file_path):
    seed_everything()
    # print(torch.cuda.is_available())
    print("File:",file_path)
    result=dict()
    for i in range(len(model_list)):
        model_path_tmp=os.path.join(logdir,model_list[i],"checkpoint",model_list[i]+".ckpt")
        if os.path.exists(model_path_tmp):
            model_tmp = MalNet.load_from_checkpoint(checkpoint_path=model_path_tmp, logdir=logdir, net=model_list[i], map_location='cpu')
            result_tmp = test_model(model_tmp, file_path, get_trans(model_list[i]))
            result[model_list[i]]=result_tmp
            print("Model:", model_list[i])
            print("Benign:", result_tmp[0])
            print("Malicious:", result_tmp[1])
            if result_tmp[0] > result_tmp[1]:
                print("Benign")
            else:
                print("Malicious")

        else:
            raise Exception("When testing, you need to assign a checkpoint path. Usage --checkpoint [path]")
    return result    


if __name__ == '__main__':
    if len(sys.argv)> 1 :
        file_path = sys.argv[1]
    EXEDetection(file_path)
