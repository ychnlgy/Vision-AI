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

    model = torch.nn.DataParallel(unet_cifar10.Unet(
        box_w=4, box_h=4
    )).to(device)
    lossf = torch.nn.L1Loss(reduction="sum")
    optim = torch.optim.AdamW(
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
                n = x.size(0)
                loss = lossf(xh.view(n, -1), x.view(n, -1))/n

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
                b = x.size(0)
                acc += lossf(xh.view(b, -1), x.view(b, -1)).item()
                n += len(x)

        print("Epoch %d test L1-loss: %.2f" % (epoch, acc/n*100.0))

        if epoch >= args.save_cycle and not epoch % args.save_cycle:
            print("Saving model to %s..." % args.save)
            torch.save(model.cpu().state_dict(), args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # System parameters
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    
    # Training cycle parameters
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--l2_reg", type=float, required=True)
    
    # Committing to disk
    parser.add_argument("--save", required=True)
    parser.add_argument("--save_cycle", type=int, required=True)
    
    args = parser.parse_args()
    main(args)
