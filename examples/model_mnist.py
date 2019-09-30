import torch

import vision_ai


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            vision_ai.models.resnet.Block(
                block=torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(32),

                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(32),
                )
            ),
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
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.linear = torch.nn.Linear(128, 10)

    def forward(self, X):
        Xh = self.cnn(X).view(X.size(0), 128)
        return self.linear(Xh)
