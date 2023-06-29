import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    t = np.array(dict[b"labels"])
    x = np.array(dict[b"data"])
  return t, x

class CifarDataset(Dataset):
    def __init__(self, phase="train"):
        self.phase = phase
        self.img = []
        self.label = []

        mean, std =  (0.49, 0.48, 0.44), (0.24, 0.24, 0.26)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if self.phase == "train":
            self.transform = transforms.Compose([
                self.transform,
                transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip()
            ])

        DATA_ROOT = "./data/cifar-10"
        if self.phase == "train":
            for i in range(1, 6):
                file_name = os.path.join(DATA_ROOT, "data_batch_%d" % i)
                label, img = unpickle(file_name)
                self.img.append(img)
                self.label.append(label)
            self.img = np.concatenate(self.img)
            self.label = np.concatenate(self.label)

        if self.phase == "valid":
            file_name = os.path.join(DATA_ROOT, "test_batch")
            self.label, self.img = unpickle(file_name)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, id):
        img = np.rollaxis(self.img[id].reshape(3, 32, 32), 0, 3)
        img = self.transform(img)
        return img, torch.tensor(self.label[id], dtype=torch.long)
