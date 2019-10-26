import torch
import torchvision
import vision_ai


class Unet_Resnet34(vision_ai.models.Unet):
    def __init__(self):
        res34_model = torchvision.models.resnet34(pretrained=True)
        modules = list(res34_model.children())
        layer1 = modules[4]
        layer1.requires_grad = False
        layer2 = modules[5]
        layer2.requires_grad = False
        layer3 = modules[6]
        layer3.requires_grad = False
        layer4 = modules[7]
        layer4.requires_grad = False
        super().__init__(
            layers=[
                torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    layer1
                ),
                layer2,
                layer3,
                layer4
            ],
            shortcuts=[
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(64 + 128),
                    torch.nn.Conv2d(64 + 128, 128, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(128 + 256),
                    torch.nn.Conv2d(128 + 256, 128, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(512+256),
                    torch.nn.Conv2d(512+256, 256, 1, bias=False)
                )
            ]
        )
        self.tail = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 2, 1, bias=False)
        )

    def forward(self, X):
        Xh = super().forward(X)
        Xh = self.tail(Xh)
        return Xh

    def forward_unet(self, X):
        Xh = super().forward(X)
        return Xh

    def forward_tail(self, X):
        Xh = self.tail(X)
        return Xh

if __name__ == '__main__':
    device = ["cpu", "cuda"][torch.cuda.is_available()]
    a = Unet_Resnet34()
    x = torch.randn(1, 3, 480, 640)
    xh = a(x)
    print(xh.size())
