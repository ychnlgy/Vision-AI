import argparse

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

import torch

import vision_ai

import unet_cifar10


def main(args):
    dataloader, testloader = vision_ai.data.cifar10.get(
        root="~/torchvision-data",
        download=False,
        batch_size=1,
        num_workers=1
    )
    
    model = unet_cifar10.Unet(
        box_w=args.box_w, box_h=args.box_h
    )
    
    model.load_state_dict(torch.load(args.save))
    
    model.train()
    
    fig, axes = pyplot.subplots(ncols=2)
    
    with torch.no_grad():
        for i, (x, _) in zip(range(args.samples), dataloader):
            xh = model(x)

            x = model.cover(x)
            x_arr = x.view(3, 32, 32).permute(1, 2, 0).numpy()
            axes[0].imshow(x_arr)
            axes[0].set_title("Before")
            xh_arr = xh.view(3, 32, 32).permute(1, 2, 0).numpy()
            axes[1].imshow(xh_arr)
            axes[1].set_title("After")
            pyplot.title("Sample %d" % i)
            fpath = "sample%d.png" % i
            print("Saving %s..." % fpath)
            pyplot.savefig(fpath, bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--box_w", type=int, required=True)
    parser.add_argument("--box_h", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--save", required=True)
    args = parser.parse_args()
    main(args)
