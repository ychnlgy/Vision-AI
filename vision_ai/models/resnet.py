import torch


class Block(torch.nn.Module):

    def __init__(self, block, shortcut=None):
        super().__init__()
        self.block = block
        self.shortcut = [shortcut, torch.nn.Sequential()][shortcut is None]

    def forward(self, X):
        return self.shortcut(X) + self.block(X)
