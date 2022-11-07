import csv
import os
import matplotlib.image
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pylab import mpl
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io

mpl.use('macosx')


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, labelsFile, rootDir, sourceTransform):

        self.data = pd.read_csv(labelsFile)
        self.rootDir = rootDir
        self.sourceTransform = sourceTransform

        return

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.rootDir + "/" + self.data['Image_path'][idx], as_gray=True)
        label = self.data['label'][idx]
        image = Image.fromarray(image)

        if self.sourceTransform:
            image = self.sourceTransform(image)

        return image, label


def generate_data():

    n = 28
    os.chdir(os.path.expanduser("/Users/alisher/IdeaProjects/Ann_practice/Scripts/data_test"))

    for i in range(2000):

        a_mat = np.random.randint(10) * np.random.rand(n, n) + np.eye(n) * np.random.randint(10)
        b_mat = np.random.randint(10) * np.dot(a_mat, a_mat.T)
        matplotlib.image.imsave('{}.png'.format(i), b_mat)

    return None


def pack():

    image_header = ["Image_path", "label"]
    image_data = []

    for i in range(20000):
        image_data.append({'Image_path': '{}.png'.format(i),
                           'label': '{}'.format(i)})

    with open('dataset_train.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=image_header)
        writer.writeheader()
        writer.writerows(image_data)

    image_data = []

    for i in range(2000):
        image_data.append({'Image_path': '{}.png'.format(i),
                           'label': '{}'.format(i)})

    with open('dataset_test.csv', 'w') as file:

        writer = csv.DictWriter(file, fieldnames=image_header)
        writer.writeheader()
        writer.writerows(image_data)

    return None


def prepare():

    custom_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = CustomDataset("dataset_train.csv", "/Users/alisher/IdeaProjects/Ann_practice/Scripts/data_train",
                                  custom_transform)
    test_dataset = CustomDataset("dataset_test.csv", "/Users/alisher/IdeaProjects/Ann_practice/Scripts/data_test",
                                 custom_transform)

    return train_dataset, test_dataset


if __name__ == "__main__":
    # generate_data()
    # pack()
    pass
