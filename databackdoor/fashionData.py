import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.io import read_image
from torch import nn
import torch.nn.functional as F

import os
import cv2
from skimage import io
import pandas as pd
from PIL import Image
from sys import argv

class customDataset(Dataset):
    def __init__(self, annotations, img_dir, flag="train", transform=None, target_transform=None):
        # super().__init__()
        assert flag in ["train", "test"]
        self.flag = flag

        self.image_labels = pd.read_csv(annotations)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[index, 0])
        # print(img_path)
        image = Image.fromarray(cv2.imread(img_path, -1), mode="L")
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, int(label)



class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        # x = F.softmax(x, dim=1)
        return x
    

class Data:
    def __init__(self):

        self.train_dataset = datasets.FashionMNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
        self.test_dataset = datasets.FashionMNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)

        self.backdoor_train = None
        self.backdoor_test = None
        self.concat_dataset = None

        self.train_loader = None
        self.test_loaderCTC = None
        self.test_loaderCTP = None
        self.test_loaderPTC = None
        self.test_loaderPTP = None

    
    def load(self, mode):
        if mode == "ctc/Accuracy":
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=100, shuffle=True)
            self.test_loaderCTC = DataLoader(dataset=self.test_dataset, batch_size=100, shuffle=True)
        elif mode == "ctp/Accuracy":
            self.train_loader = DataLoader(dataset=self.concat_dataset, batch_size=100, shuffle=True)
            self.test_loaderCTP = DataLoader(dataset=self.test_dataset, batch_size=100, shuffle=True)
        elif mode == "ptc/Accuracy":
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=100, shuffle=True)
            self.test_loaderPTC = DataLoader(dataset=self.backdoor_test, batch_size=100, shuffle=True)
            self.clean_test = DataLoader(dataset=self.test_dataset, batch_size=100, shuffle=True)
        elif mode == "ptp/ASR":
            self.train_loader = DataLoader(dataset=self.concat_dataset, batch_size=100, shuffle=True)
            self.test_loaderPTP = DataLoader(dataset=self.backdoor_test, batch_size=100, shuffle=True)
        elif mode == "P-model":
            self.train_loader = DataLoader(dataset=self.concat_dataset, batch_size=100, shuffle=True)
            self.test_loaderCTP = DataLoader(dataset=self.test_dataset, batch_size=100, shuffle=True)
            self.test_loaderPTC = DataLoader(dataset=self.backdoor_test, batch_size=100, shuffle=True)
            self.test_loaderPTP = DataLoader(dataset=self.backdoor_test, batch_size=100, shuffle=True)
        elif mode == "C-model":
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=100, shuffle=True)
            self.test_loaderCTC = DataLoader(dataset=self.test_dataset, batch_size=100, shuffle=True)



    # uncomplete
    def loadPoison(self):
        self.backdoor_train = customDataset(annotations="data/PoisonedMNIST/label.csv", img_dir="data/PoisonedMNIST/img", transform=transforms.ToTensor(), flag="train")
        self.backdoor_test = customDataset(annotations="data/PoisonedMNIST/testlabel.csv", img_dir="data/PoisonedMNIST/test", transform=transforms.ToTensor(), flag="test")
        self.concat_dataset = ConcatDataset([self.train_dataset, self.backdoor_train])

    def getTestData(self, mode):
        if mode == "ctc/Accuracy":
            return self.test_loaderCTC
        elif mode == "ctp/Accuracy":
            return self.test_loaderCTP
        elif mode == "ptc/Accuracy":
            return self.test_loaderPTC
        elif mode == "ptp/ASR":
            return self.test_loaderPTP
        elif mode == "P-model":
            return self.test_loaderPTP
        elif mode == "C-model":
            return self.test_loaderCTC