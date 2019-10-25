import torch
import torchvision
import vision_ai


class Unet_Resnet50(vision_ai.models.Unet):
    def __init__(self):
        res50_model = torchvision.models.resnet50(pretrained=True)
        modules = list(res50_model.children())
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
                    torch.nn.BatchNorm2d(512 + 256),
                    torch.nn.Conv2d(512 + 256, 256, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(1024 + 512),
                    torch.nn.Conv2d(1024 + 512, 512, 1, bias=False)
                ),
                torch.nn.Sequential(
                    torch.nn.BatchNorm2d(2048+1024),
                    torch.nn.Conv2d(2048+1024, 1024, 1, bias=False)
                )
            ]
        )
        self.tail = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 2, 1, bias=False),
            torch.nn.Tanh()
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
    a = Unet_Resnet50()
    x = torch.randn(1, 3, 120, 200)
    xh = a.forward_unet(x)
    print(xh.size())
    xh = a.forward_tail(xh)
    print(xh.size())
