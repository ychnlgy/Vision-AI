import torch

import vision_ai


class Model(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 128, 3, padding=1),
            vision_ai.models.resnet.Block(
                block=torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(128),

                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(128),
                )
            ),
            vision_ai.models.resnet.Block(
                block=torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 128, 3, padding=1, bias=False, stride=2),  # 32 -> 16
                    torch.nn.BatchNorm2d(128),

                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(128),
                ),
                shortcut=torch.nn.Conv2d(128, 128, 1, stride=2, bias=False)
            ),
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
            ),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.linear = torch.nn.Linear(256, 10)

    def forward(self, X):
        Xh = self.cnn(X).view(X.size(0), 256)
        return self.linear(Xh)
