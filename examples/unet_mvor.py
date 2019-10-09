import torch

import vision_ai


class Unet_Mvor(vision_ai.models.Unet):
    def __init__(self):
        super().__init__(
            layers=[
                # N, 32, 32, 32
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.Sequential(
                    vision_ai.models.resnet.Block(
                        block=torch.nn.Sequential(
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(64, 128, 3, padding=1, bias=False, stride=2),
                            torch.nn.BatchNorm2d(128),

                            torch.nn.ReLU(),
                            torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(128),
                        ),
                        shortcut=torch.nn.Conv2d(64, 128, 1, stride=2, bias=False)
                    )
                ),
                # N, 64, 16, 16
                vision_ai.models.resnet.Block(
                    block=torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(128, 256, 3, padding=1, bias=False, stride=2),  # 32 -> 16
                        torch.nn.BatchNorm2d(256),

                        torch.nn.ReLU(),
                        torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        torch.nn.BatchNorm2d(256),
                    ),
                    shortcut=torch.nn.Conv2d(128, 256, 1, stride=2, bias=False)
                ),
                # N, 128, 8, 8
                vision_ai.models.resnet.Block(
                    block=torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(256, 512, 3, padding=1, bias=False, stride=2),  # 16 -> 8
                        torch.nn.BatchNorm2d(512),

                        torch.nn.ReLU(),
                        torch.nn.Conv2d(512, 512, 3, padding=1, bias=False),
                        torch.nn.BatchNorm2d(512),
                    ),
                    shortcut=torch.nn.Conv2d(256, 512, 1, stride=2, bias=False)
                ),
                # N, 256, 4, 4
                vision_ai.models.resnet.Block(
                    block=torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(512, 1024, 3, padding=1, bias=False, stride=2),  # 16 -> 8
                        torch.nn.BatchNorm2d(1024),

                        torch.nn.ReLU(),
                        torch.nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
                        torch.nn.BatchNorm2d(1024),
                    ),
                    shortcut=torch.nn.Conv2d(512, 1024, 1, stride=2, bias=False)
                )
            ],
            shortcuts=[
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(128 + 64),
                    torch.nn.Conv2d(128 + 64, 128, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(256 + 128),
                    torch.nn.Conv2d(256 + 128, 128, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(512 + 256),
                    torch.nn.Conv2d(512 + 256, 256, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(1024+512),
                    torch.nn.Conv2d(1024+512, 512, 1, bias=False)
                )
            ]
        )
        self.tail = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, X):
        Xh = super().forward(X)
        Xh = self.tail(Xh)
        return Xh

if __name__ == '__main__':
    a = Unet_Mvor().to('cuda')
    x = torch.randn((2, 3, 480, 640)).to('cuda')
    print(a(x))
