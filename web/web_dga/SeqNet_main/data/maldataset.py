import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from SeqNet_main.utils.utils import get_all_file_path
from pytorch_lightning.core.datamodule import LightningDataModule

class MalwareDataset(Dataset):
    def __init__(self, mal_paths : list, norm_paths : list, trans : callable):
        super().__init__()
        self.mal = mal_paths
        self.norm = norm_paths
        self.transform = trans

    def __len__(self):
        return len(self.mal) + len(self.norm)

    def __getitem__(self, item):
        if item >= len(self.mal) :
            item -= len(self.mal)
            with open(self.norm[item], "rb") as f:
                data = f.read()
                data = self.transform(data)
                return data, torch.tensor(0)
        else :
            with open(self.mal[item], "rb") as f:
                data = f.read()
                data = self.transform(data)
                return data, torch.tensor(1)

class MalDataModule(LightningDataModule):
    def __init__(self, mal_path : str, norm_path : str, trans : callable, batch_size = 50, val_num = 100, val_mal=None, val_norm=None):
        super().__init__()
        self.mal = mal_path
        self.norm = norm_path
        self.batch_size = batch_size
        self.val_num = val_num
        self.train_data = []
        self.val_data = []
        self.val_mal = val_mal
        self.val_norm = val_norm
        self.trans = trans

    def setup(self, stage) :
        assert os.path.exists(self.mal)
        assert os.path.exists(self.norm)
        if self.val_mal is not None:
            assert os.path.exists(self.val_mal)
            self.train_data = [get_all_file_path(self.mal)]
            self.val_data = [get_all_file_path(self.val_mal)]
        else:
            self.train_data = [get_all_file_path(self.mal)[:-self.val_num]]
            self.val_data = [get_all_file_path(self.mal)[-self.val_num:]]

        if self.val_norm is not None:
            assert os.path.exists(self.val_norm)
            self.train_data.append(get_all_file_path(self.norm))
            self.val_data.append(get_all_file_path(self.val_norm))
        else:
            self.train_data.append(get_all_file_path(self.norm)[:-self.val_num])
            self.val_data.append(get_all_file_path(self.norm)[-self.val_num:])

    def train_dataloader(self) :
        dataset = MalwareDataset(self.train_data[0], self.train_data[1], self.trans)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, num_workers=10)
        return dataloader

    def val_dataloader(self) :
        dataset = MalwareDataset(self.val_data[0], self.val_data[1], self.trans)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=10)
        return dataloader

    def test_dataloader(self) :
        return self.val_dataloader()