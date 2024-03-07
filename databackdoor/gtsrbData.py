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
        image = Image.fromarray(cv2.imread(img_path, -1), mode="RGB")
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, int(label)
    

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250*2*2, 350)
        self.fc2 = nn.Linear(350, 43)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )

        
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


    def forward(self, x):
        x = self.stn(x)

        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        x = self.conv_drop(x)
        x = x.view(-1, 250*2*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Data:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                (0.2724, 0.2608, 0.2669))
        ])

        self.train_dataset = datasets.GTSRB(root="data/", split="train", transform=self.transform, download=True)
        self.test_dataset = datasets.GTSRB(root="data/", split="test", transform=self.transform, download=True)

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
        self.backdoor_train = customDataset(annotations="data/PoisonedGTSRB/label.csv", img_dir="data/PoisonedGTSRB/img", transform=self.transform, flag="train")
        self.backdoor_test = customDataset(annotations="data/PoisonedGTSRB/testlabel.csv", img_dir="data/PoisonedGTSRB/test", transform=self.transform, flag="test")
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


