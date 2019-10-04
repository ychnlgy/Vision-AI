import argparse

import torch

import vision_ai
import visualize_cifar10_unet

def main(args):
    
    unet = unet_cifar10.Unet()
    unet.load_state_dict(torch.load(args.prev_save, map_location="cpu"))
    
    dataset, testset = vision_ai.data.cifar10.get(
        root="~/torchvision-data",
        download=0,
        batch_size=128,
        num_workers=8
    )
    
    for x, y in dataset:
        xh = unet(x)
        visualize_cifar10_unet.visualize(x, xh, x, n=10)
        break
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev_save", required=True)
    main(parser.parse_args())
