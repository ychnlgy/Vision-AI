import torch

import vision_ai


class Unet(vision_ai.models.Unet):

    def __init__(self):
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
                    torch.nn.BatchNorm2d(128 + 32),
                    torch.nn.Conv2d(128 + 32, 128, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(128 + 64),
                    torch.nn.Conv2d(128 + 64, 128, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(256 + 128),
                    torch.nn.Conv2d(256 + 128, 128, 1, bias=False)
                )
            ]
        )
        self.tail = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 3, 1, bias=False),
            torch.nn.Tanh()
        )
    
    def forward(self, X):
        Xh = super().forward(X)
        Xh = self.tail(Xh)
        return Xh
