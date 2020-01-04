import argparse

import torch
import tqdm

import vision_ai

import model_cifar10
import model_mnist


def main(args):
    device = ["cpu", "cuda"][torch.cuda.is_available()]
    
    data_getter, Model = {
        "mnist": (vision_ai.data.mnist.get, model_mnist.Model),
        "cifar10": (vision_ai.data.cifar10.get, model_cifar10.Model)
    }[args.dataset]
    
    dataloader, testloader = data_getter(
        root="~/torchvision-data",
        download=True,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    model = Model().to(device)
    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2_reg
    )
    
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs
    )

    for epoch in range(1, args.epochs+1):
        
        model.train()
        
        with tqdm.tqdm(dataloader, ncols=80) as bar:
        
            for x, y in bar:
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # System parameters
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    
    # Training cycle parameters
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--l2_reg", type=float, required=True)
    
    # Data parameters
    parser.add_argument("--dataset", required=True)
    
    args = parser.parse_args()
    main(args)
