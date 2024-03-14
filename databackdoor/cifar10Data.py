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
        image = Image.fromarray(cv2.imread(img_path, -1), mode="RGB")
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, int(label)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])
    

class Data:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = datasets.CIFAR10(root="data/", train=True, transform=self.transform, download=True)
        self.test_dataset = datasets.CIFAR10(root="data/", train=False, transform=self.transform_test, download=True)

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
        self.backdoor_train = customDataset(annotations="data/PoisonedCIFAR10/label.csv", img_dir="data/PoisonedCIFAR10/img", transform=self.transform, flag="train")
        self.backdoor_test = customDataset(annotations="data/PoisonedCIFAR10/testlabel.csv", img_dir="data/PoisonedCIFAR10/test", transform=self.transform_test, flag="test")
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