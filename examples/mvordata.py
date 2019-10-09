import os
import json
import cv2
import numpy as np
import tqdm
import torch
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision

# root is the path to MVOR dataset
def getAllImagePaths(root):
    imagePaths = []
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if os.path.isdir(path):
            subdir_imagepaths = getAllImagePaths(path)
            imagePaths += subdir_imagepaths
        elif os.path.isfile(path) and os.path.splitext(path)[1] == '.png':
            imagePaths.append(path)
    return imagePaths


def getImagePathFromJson(path_to_jsonfile, path_to_mvor_dataset):
    path_list = []
    if path_to_jsonfile:
        with open(path_to_jsonfile, 'r') as f:
            datastore = json.load(f)
            for images in datastore['multiview_images']:
                for camera in images['images']:
                    file_name = camera['file_name']
                    final_path = os.path.join(path_to_mvor_dataset, file_name)
                    file_id = camera['id']
                    path_list.append((final_path, file_id))
    return path_list


def splitset(data, portion):
    random.shuffle(data)
    n = int(float(portion)*len(data))
    test_data = data[:n]
    train_data = data[n:]
    return train_data, test_data


# class Mvordatapath:
#     def __init__(self, root, path_to_json_file):
#         self.root = root
#         self.path_to_json_file = path_to_json_file
#         self.allImagePath = getImagePathFromJson(self.path_to_json_file)
#
#     def getAllColorImagePath(self):
#         return [path for path in self.allImagePath if path.split('/')[-2] == 'color']
#
#     def getAllDepthImagePath(self):
#         return [path for path in self.allImagePath if path.split('/')[-2] == 'depth']


# convert data path into dataset arrays
class Mvordata:
    def __init__(self, path_to_jsonfile, path_to_mvor_dataset, image_paths):
        self.path_to_jsonfile = path_to_jsonfile
        self.image_paths = image_paths

    def to_dataset(self):
        with open(self.path_to_jsonfile, 'r') as f:
            datastore = json.load(f)
            data_list = []
            label_list = []
            print('Processing Data: ')
            for i in tqdm.tqdm(range(len(self.image_paths))):
                data = self._get_data(self.image_paths[i][0])
                bbox = self._get_bbox(datastore, self.image_paths[i][1])
                label = np.zeros((data.shape[1:]), np.uint8)
                if len(bbox) > 0:
                    for x, y, w, h in bbox:
                        label[int(y):int(y+h), int(x):int(x+w)] = 1
                data_list.append(data)
                label_list.append(label)
            print("Data Processing Done !")
        return data_list, label_list

    def _get_data(self, path):
        # img = Image.open(path)
        # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # # C x H x W
        # data = transform(img)
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        data = np.transpose(data, (2, 0, 1))
        return data

    def _get_bbox(self, datastore, id):
        bbox = []
        for image in datastore['annotations']:
            if image['image_id'] == id:
                bbox.append(image['bbox'])
        return bbox


def create_loaders(batch_size, num_workers, X_train, X_test, y_train, y_test, shuffle_test=False):
    # X_train, X_test, y_train, y_test = train_test_split(datalist, labellist, test_size=test_size, random_state=42)
    # print(np.array(X_train).shape, np.array(X_test).shape, np.array(y_train).shape, np.array(y_test).shape)
    train_data = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(X_train)),
                                                torch.LongTensor(np.array(y_train)))
    test_data = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(X_test)), torch.LongTensor(np.array(y_test)))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers
    )
    # n = 0
    # for x, y in train_loader:
    #     print(x.size(), y.size())
    #     x1 = x[0].numpy()
    #     y1 = y[0].numpy()
    #     print(y1.sum())
    #     n += 1
    #     if n == 5:
    #         break
    print('Data Loader Created')
    return train_loader, test_loader

if __name__ == '__main__':
    DATAPATH = "../../../mvor-master/dataset"
    JsonFile = "../../../mvor-master/annotations/camma_mvor_2018.json"
    all_path = getImagePathFromJson(JsonFile, DATAPATH)
    train_path, test_path = split_data(all_path, 0.2)

    train_data = Mvordata(JsonFile, DATAPATH, train_path)
    train_data_list, train_label_list = train_data.to_dataset()

    test_data = Mvordata(JsonFile, DATAPATH, test_path)
    test_data_list, test_label_list = test_data.to_dataset()

    train_loader, test_loader = create_loaders(16, 0, train_data_list, test_data_list, train_label_list, test_label_list)




