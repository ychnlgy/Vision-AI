import torch

import unet_cifar10


class Classifier(torch.nn.Module):
    
    def __init__(self, unet, tune, tail):
        super().__init__()
        self.unet = unet
        self.tune = tune
        self.tail = tail
    
    def forward(self, X):
        if tune:
            X = self.unet(X)
        else:
            with torch.no_grad():
                X = self.unet(X).clone().detach()
        return self.tail(X)
