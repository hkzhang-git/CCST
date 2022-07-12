import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class GIST(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feat = self.data[idx]
        feat = (feat-0.0696)/0.0468
        return torch.from_numpy(feat)

class Deep1M(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feat = self.data[idx]
        feat = (feat+0.0009)/0.0621
        return torch.from_numpy(feat)

dataset_dict = {
    'GIST1M': GIST,
    'Deep1M': Deep1M,
}



