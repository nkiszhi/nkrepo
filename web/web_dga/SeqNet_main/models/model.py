import os, json, torch
import torch.nn as nn
import pytorch_lightning as pl
from SeqNet_main.models.regionnet.network import *
from SeqNet_main.models.malconv_gct.network import MalConv
from torchvision.models import MobileNetV2, resnet18, mobilenet_v3_large, mobilenet_v3_small
from SeqNet_main.utils.test_val import output_process
from SeqNet_main.models.seqnet.network import *
from SeqNet_main.models.malconv_gct.network import MalConvGCT

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = MobileNetV2(num_classes=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        if not self.training:
            x = self.softmax(x)
        return x
    
class MobileNetLarge(nn.Module):
    def __init__(self):
        super(MobileNetLarge, self).__init__()
        self.model = mobilenet_v3_large(num_classes=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        if not self.training:
            x = self.softmax(x)
        return x

class MobileNetSmall(nn.Module):
    def __init__(self):
        super(MobileNetSmall, self).__init__()
        self.model = mobilenet_v3_small(num_classes=2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.model(x)
        if not self.training:
            x = self.softmax(x)
        return x
    

class MalNet(pl.LightningModule):
    def __init__(self, logdir, net="malnet"):
        super(MalNet, self).__init__()
        self.logdir = os.path.join(logdir, "log")
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)

        self.model = None
        if net == "regionnet":
            self.model = RegionNet(input_size=(512, 512, 1))
        elif net == "regionnetconv":
            self.model = RegionNetConv(input_size=(512, 512, 1))
        elif net == "mobilenet":
            self.model = MobileNet()
        elif net == "mobilenetlarge":
            self.model = MobileNetLarge()
        elif net == "mobilenetsmall":
            self.model = MobileNetSmall()
        elif net == "malconv":
            self.model = MalConv()
        elif net == "seqnetdeep":
            self.model = SequenceNetDeeper()
        elif net == "seqnetshal":
            self.model = SequenceNetShallow()
        elif net == "seqnetconv":
            self.model = SequenceNetConv()
        elif net == "seqnetfool":
            self.model = SequenceNetFool()
        elif "seqnet" in net:
            self.model = SequenceNet()  # 136 K
        elif net == "malconv2":
            self.model = MalConvGCT()
        else:
            raise Exception("No such model:", net)

        self.loss = nn.CrossEntropyLoss() # 统一使用交叉熵损失函数

    def get_loss(self, x, y):
        return self.loss(x, y)
    # 继承pytorch_lighting.LightningModule
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters()) # 统一使用Adam优化器

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        loss = self.get_loss(x, y)
        self.log("Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        loss = self.get_loss(x, y)
        return x, y, loss

    def validation_epoch_end(self, outputs) -> None:
        result, _, _, _ = output_process(outputs)
        print(result)
        with open(os.path.join(self.logdir, "log.json"), "a") as f:
            f.write(json.dumps(result) + "\n")
        self.log_dict(result)
