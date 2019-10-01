import random

import torch
import torch.utils.data


class CoverDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, box):
        self.dataset = dataset
        if type(box) is int:
            self.box_w = box
            self.box_h = box
        else:
            self.box_w, self.box_h = box

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        x, y = self.dataset[key]
        C, W, H = x.size()
        i = random.randint(0, W - self.box_w)
        j = random.randint(0, H - self.box_h)
        xc = x.clone()
        xc[:, i:i + self.box_w, j:j + self.box_h] = 0
        return x, xc, y
