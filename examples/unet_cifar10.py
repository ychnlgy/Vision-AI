import random

import torch

import vision_ai


class Unet(vision_ai.models.Unet):

    def __init__(self, box_w, box_h):
        super().__init__(
            layers = [
                # N, 32, 32, 32
                torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, 3, padding=1),
                    vision_ai.models.resnet.Block(
                        block=torch.nn.Sequential(
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(32),

                            torch.nn.ReLU(),
                            torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(32),
                        )
                    )
                ),
                # N, 64, 16, 16
                vision_ai.models.resnet.Block(
                    block=torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(32, 64, 3, padding=1, bias=False, stride=2),  # 32 -> 16
                        torch.nn.BatchNorm2d(64),

                        torch.nn.ReLU(),
                        torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
                        torch.nn.BatchNorm2d(64),
                    ),
                    shortcut=torch.nn.Conv2d(32, 64, 1, stride=2, bias=False)
                ),
                # N, 128, 8, 8
                vision_ai.models.resnet.Block(
                    block=torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(64, 128, 3, padding=1, bias=False, stride=2),  # 16 -> 8
                        torch.nn.BatchNorm2d(128),

                        torch.nn.ReLU(),
                        torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
                        torch.nn.BatchNorm2d(128),
                    ),
                    shortcut=torch.nn.Conv2d(64, 128, 1, stride=2, bias=False)
                ),
                # N, 256, 4, 4
                vision_ai.models.resnet.Block(
                    block=torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(128, 256, 3, padding=1, bias=False, stride=2),  # 16 -> 8
                        torch.nn.BatchNorm2d(256),

                        torch.nn.ReLU(),
                        torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        torch.nn.BatchNorm2d(256),
                    ),
                    shortcut=torch.nn.Conv2d(128, 256, 1, stride=2, bias=False)
                )
            ],
            shortcuts = [
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128 + 32, 128, 1, bias=False),
                    torch.nn.BatchNorm2d(128)
                ),
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128 + 64, 128, 1, bias=False),
                    torch.nn.BatchNorm2d(128)
                ),
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(256 + 128, 128, 1, bias=False),
                    torch.nn.BatchNorm2d(128)
                )
            ]
        )
        self.tail = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 3, 1, bias=False),
            torch.nn.Tanh()
        )
        self.box_w = box_w
        self.box_h = box_h
    
    def forward(self, X):
        if self.training:
            X = self.cover(X)
        Xh = super().forward(X)
        return self.tail(Xh)
    
    def cover(self, X):
        X = X.clone()
        N, C, W, H = X.size()
        for i in range(N):
            x = random.randint(0, W - self.box_w)
            y = random.randint(0, H - self.box_h)
            X[i, :, x:x + self.box_w, y:y + self.box_h] = 0.0
        return X
