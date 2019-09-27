import argparse

import torch
import tqdm

import vision_ai


def main(args):
    device = ["cpu", "cuda"][torch.cuda.is_available()]
    
    dataloader, testloader = vision_ai.data.mnist.get(
        root="~/torchvision-data",
        download=True,
        batch_size=128,
        num_workers=4
    )

    model = Model().to(device)
    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(
        model.parameters(),
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs
    )

    for epoch in range(1, args.epochs+1):
        
        model.train()
        
        with tqdm.tqdm(dataloader, ncols=80) as bar:
        
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                yh = model(x)
                loss = lossf(yh, y)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                bar.set_description("Loss: %.4f" % loss.item())
        
        sched.step()
        
        model.eval()
        
        acc = n = 0.0
        
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                yh = model(x)
                vals, args = yh.max(dim=1)
                acc += (args == y).sum().item()
                n += len(y)
        
        print("Epoch %d test accuracy: %.2f%%" % (epoch, acc/n*100.0))

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
                shortcut=vision_ai.models.resnet.Shortcut(32, 64)
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
                shortcut=vision_ai.models.resnet.Shortcut(64, 128)
            ),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(8)
        )
        self.linear = torch.nn.Linear(128, 10)

    def forward(self, X):
        Xh = self.cnn(X).view(X.size(0), 128)
        return self.linear(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    
    args = parser.parse_args()
    main(args)
