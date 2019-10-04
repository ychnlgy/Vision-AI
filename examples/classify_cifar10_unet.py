import argparse
import sys

import torch
import tqdm

import unet_cifar10
import vision_ai

class Classifier(torch.nn.Module):
    
    def __init__(self, unet, tune, tail):
        super().__init__()
        self.unet = unet
        self.unet.tail = torch.nn.Sequential()
        self.tune = tune
        self.tail = tail
    
    def forward(self, X):
        N = len(X)
        if self.tune:
            X = self.unet(X)
        else:
            with torch.no_grad():
                X = self.unet(X).clone().detach()
        classification = self.tail(X)
        return classification.view(N, classification.size(1))


def main(args):
    
    device = ["cpu", "cuda"][torch.cuda.is_available()]
    
    dataset, testset = vision_ai.data.cifar10.get(
        root="~/torchvision-data",
        download=args.download,
        batch_size=128,
        num_workers=8
    )
    
    unet = unet_cifar10.Unet()
    unet.load_state_dict(torch.load(args.prev_save, map_location="cpu"))
    
    cpu_model = Classifier(
        unet,
        args.tune,
        tail=torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 10, 1),
            torch.nn.AdaptiveAvgPool2d(1)
        )
    )
    
    model = torch.nn.DataParallel(cpu_model).to(device)
    
    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.l2_reg
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        
        model.train()
        
        avg = vision_ai.utils.MovingAvg(0.95)
        
        with tqdm.tqdm(dataset, ncols=80) as bar:
            for x, y in bar:
                x = x.to(device)
                y = y.to(device)
                yh = model(x)
                loss = lossf(yh, y)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                avg.update(loss.item())
                bar.set_description("Epoch %d loss: %.4f" % (epoch, avg.peek()))
        
        model.eval()
        
        acc = n = 0.0
        
        with torch.no_grad():
            
            for x, y in testset:
                x = x.to(device)
                y = y.to(device)
                yh = model(x)
                acc = (yh.max(dim=1)[1] == y).long().sum().item()
                n += len(y)
        
        sys.stderr.write("Test accuracy: %.2f%%\n" % (acc/n*100.0))
        sys.stderr.flush()
    
    sys.stderr.write("Saving to %s..." % args.save)
    sys.stderr.flush()
    
    torch.save(cpu_model.state_dict(), args.save)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--download", type=int, default=0)
    
    parser.add_argument("--prev_save", required=True)
    parser.add_argument("--tune", type=int, required=True)
    parser.add_argument("--save", required=True)
    
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--l2_reg", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    
    args = parser.parse_args()
    
    main(args)
