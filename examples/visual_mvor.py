import os

import numpy
import torch
import argparse
from mvordata import *
from unet_mvor import *
from resnet34 import *
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


def main(args):
        device = ["cpu", "cuda"][torch.cuda.is_available()]
        model_path = args.model
        models = [Unet_Mvor(), Unet_Resnet34()]
        model = models[int(args.model_type)]
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        path = args.test_path
        JsonFile = args.json_file
        
        fig, axes = plt.subplots(ncols=4, sharey=True, figsize=(18, 12))
        
        with torch.no_grad():
                with open(path, 'rb') as f:
                        test_path = pickle.load(f)
                        print(test_path[0])
                        test_data = Mvordata(JsonFile, test_path)
                        test_data_list, test_label_list = test_data.to_dataset()
                        n = 0
                        acc = 0
                        for i in tqdm.tqdm(range(len(test_data_list)), desc='Calculating IoU'):
                                x = test_data_list[i]
                                x = torch.tensor(x).float().view(1, 3, 480, 640)
                                y = test_label_list[i]
                                y = torch.tensor(y).float()
                                x = x.to(device)
                                y = y.to(device)
                                xh = model(x)
                                xh = xh[0]
                                xh = xh.permute(1, 2, 0)
                                xh1 = xh[:, :, 1]
                                xh0 = xh[:, :, 0]
                                xh_box = xh1 > xh0
                                intersection = xh_box.float() * y
                                intersection_sum = intersection.sum()
                                union_sum = xh_box.sum() + y.sum() - intersection_sum
                                if intersection_sum > 0:
                                        n += 1
                                        acc += intersection_sum / union_sum
                                if i < args.num_image_samples:
                                        for ax in axes:
                                            ax.cla()
                                        im = x.cpu()[0].permute(1, 2, 0).numpy()/255.0
                                        
                                        axes[0].set_title("Input")
                                        axes[0].imshow(im)
                                        
                                        axes[1].set_title("Annotation")
                                        axes[1].imshow(y.cpu().numpy(), cmap="hot", interpolation="nearest")
                                        
                                        axes[2].set_title("Model output")
                                        model_output = xh_box.cpu().numpy()
                                        axes[2].imshow(model_output, cmap="hot", interpolation="nearest")
                                        
                                        axes[3].set_title("Depth refinement")
                                        path = test_data.image_paths[i][0].replace("color", "depth")
                                        assert os.path.isfile(path)
                                        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                                        cut = cutout_human(depth, model_output, args.cut_thickness)
                                        axes[3].imshow(cut, cmap="hot", interpolation="nearest")
                                        
                                        plt.savefig("sample%d-iou%.4f.png" % (i, intersection_sum/union_sum), bbox_inches="tight")
                                elif args.num_image_samples > 0:
                                    break
                        print(n, acc/n)

                # image_path = test_path[i+99]
                # print(image_path)
                # data = cv2.imread(image_path[0], cv2.IMREAD_UNCHANGED)
                # data = np.transpose(data, (2, 0, 1))
                # data = torch.tensor(data).float().view(1, 3, 480, 640)
                # start_time = time.time()
                # xh = model(data)
                # print("--- %s seconds ---" % (time.time() - start_time))
                # xh = xh[0]
                # xh = xh.permute(1, 2, 0)
                # xh1 = xh[:, :, 1]
                # xh0 = xh[:, :, 0]
                # xh_box = xh1 > xh0
                # if args.evaluate:
                #       print(xh_box.size())
                        #
                # f = plt.figure()
                # f.add_subplot(1,2, cutout_human1)
                # imgplot = mpimg.imread(image_path[0])
                # plt.imshow(imgplot)
                # f.add_subplot(1,2, 2)
                # plt.imshow(xh_box)
                # plt.show(block=True)

def cutout_human(depth, pred_box, thickness):
    med = numpy.median(depth[pred_box.astype(bool)])
    return (numpy.abs(depth * pred_box.astype(depth.dtype) - med) < thickness).astype(float)

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_image_samples", type=int, default=0)
        parser.add_argument("--cut_thickness", type=int, default=250)
        parser.add_argument("--model_type")
        parser.add_argument("--model")
        parser.add_argument("--test_path")
        parser.add_argument("--json_file", default="../../mvor-master/annotations/camma_mvor_2018.json")
        args = parser.parse_args()
        main(args)
        
