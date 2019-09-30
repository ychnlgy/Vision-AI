import argparse

import torch
import tqdm

import vision_ai

import unet_cifar10


def main(args):
    device = ["cpu", "cuda"][torch.cuda.is_available()]

    dataloader, testloader = vision_ai.data.cifar10.get(
        root="~/torchvision-data",
        download=True,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    model = unet_cifar10.Unet().to(device)
    lossf = torch.nn.MSELoss()
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
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
                xh = model(x)
                loss = lossf(xh, x)

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
                xh = model(x)
                acc += lossf(xh, x).item()
                n += 1

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
    
    args = parser.parse_args()
    main(args)
