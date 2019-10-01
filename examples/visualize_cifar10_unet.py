import argparse

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

import torch

import vision_ai

import unet_cifar10


FIG = None
AXE = None


def main(args):
    dataloader, testloader = vision_ai.data.cifar10.get(
        root="~/torchvision-data",
        download=False,
        batch_size=args.samples,
        num_workers=1
    )
    
    model = unet_cifar10.Unet(
        box_w=args.box_w, box_h=args.box_h
    )
    
    model.load_state_dict(torch.load(args.save, map_location="cpu"))
    
    model.eval()
    
    with torch.no_grad():
        for x, _ in dataloader:
            xh = model(x)
            visualize(x, xh, model)
            break
            


def visualize(X, Xh, model, n):
    global FIG
    global AXE
    
    if FIG is None:
        FIG, AXE = pyplot.subplots(ncols=2)

    for i, x, xh in zip(range(n), X, Xh):
        x = model.cover(x)
        x_arr = x.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()
        AXE[0].imshow(x_arr)
        AXE[0].set_title("Before")
        xh_arr = xh.view(3, 32, 32).permute(1, 2, 0).cpu().numpy()
        AXE[1].imshow(xh_arr)
        AXE[1].set_title("After")
        pyplot.suptitle("Sample %d" % i)
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
