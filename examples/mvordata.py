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
import pickle


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


def save_K_sample_fromday4(path_to_jsonfile, path_to_mvor_dataset, k, test_path, train_path):
    all_path = getImagePathFromJson(path_to_jsonfile, path_to_mvor_dataset)
    k_samples = []
    left = all_path.copy()
    for path in all_path:
        if k <= 0:
            break
        else:
            if 'day4' in path[0]:
                k_samples.append(path)
                left.remove(path)
                k -= 1
    with open(train_path, 'wb') as f_train:
        pickle.dump(left, f_train, pickle.HIGHEST_PROTOCOL)
    with open(test_path, 'wb') as f_test:
        pickle.dump(k_samples, f_test, pickle.HIGHEST_PROTOCOL)
    print('{} samples from day 4 created and saved as pickle'.format(len(k_samples)))


def getImagePathFromJson(path_to_jsonfile, path_to_mvor_dataset):
    path_list = []
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


class Mvordata:
    def __init__(self, path_to_jsonfile, image_paths):
        self.path_to_jsonfile = path_to_jsonfile
        self.image_paths = image_paths

    def to_dataset(self):
        with open(self.path_to_jsonfile, 'r') as f:
            datastore = json.load(f)
            data_list = []
            label_list = []
            for i in tqdm.tqdm(range(len(self.image_paths)), desc='Processing Data'):
                data = self._get_data(self.image_paths[i][0])
                bbox = self._get_bbox(datastore, self.image_paths[i][1])
                label = np.zeros((data.shape[1:]), np.uint8)
                if len(bbox) > 0:
                    for x, y, w, h in bbox:
                        label[int(y):int(y+h), int(x):int(x+w)] = 1
                data_list.append(data)
                label_list.append(label)
        return data_list, label_list

    def _get_data(self, path):
        # data = np.array(Image.open(path))
        # data = np.transpose(data, (2, 0, 1))
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
    print('Data Loader Created')
    return train_loader, test_loader


if __name__ == '__main__':
    # DATAPATH = "../../mvor-master/dataset"
    # JsonFile = "../../mvor-master/annotations/camma_mvor_2018.json"
    # all_path = getImagePathFromJson(JsonFile, DATAPATH)
    # train_path, test_path = splitset(all_path, 0.2)
    #
    # train_data = Mvordata(JsonFile, train_path)
    # train_data_list, train_label_list = train_data.to_dataset()
    #
    # test_data = Mvordata(JsonFile, test_path)
    # test_data_list, test_label_list = test_data.to_dataset()
    #
    # train_loader, test_loader = create_loaders(2, 0, train_data_list, test_data_list, train_label_list, test_label_list)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../../mvor-master/dataset")
    parser.add_argument("--json_file", default="../../mvor-master/annotations/camma_mvor_2018.json")
    args = parser.parse_args()

    save_K_sample_fromday4(args.json_file, args.data_path, 500, 'day4_test.pickle', 'day4_train.pickle')



