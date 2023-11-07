#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import torch.nn.functional as F

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import os

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
        return x


def predict(test):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("./model/MNIST_para.pth"))
    torch.no_grad()
    model.eval()

    pred_test_prob = []
    highestLabel = []
    trueLabel = []

    for i, y in test:
        output = model(i)
        probs = F.softmax(output, dim=1)
        top_probs, top_labels = torch.topk(probs, k=10)
        temp = []
        for j in range(10):
            # print(f"Prob: {top_probs[0][j].item():.4f}  Label: {top_labels[0][j]}")
            temp.append(top_probs[0][j].item())
        # print("-----------------------------------")
        highestLabel.append(top_labels[0][0].item())
        pred_test_prob.append(temp)
        trueLabel.append(y.item())


    return pred_test_prob, highestLabel, trueLabel
    

        

if __name__ == "__main__":
    test_dataset = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)
    backdoor_test = customDataset(annotations="data/PoisonedMNIST/testlabel.csv", img_dir="data/PoisonedMNIST/test", transform=transforms.ToTensor(), flag="test")

    sample = [i for i in range(10)]

    X_test = []
    Y_test = []

    for i in sample:
        X = backdoor_test[i][0]
        X_test.append(X)
        Y = backdoor_test[i][1]
        Y_test.append(Y)

    part_Test_data = [(x,y) for x, y in zip(X_test, Y_test)]

    # print(part_Test_data)

    test_data = DataLoader(dataset=part_Test_data, batch_size=1, shuffle=True)

    pred_test_prob, highestLabel, trueLabel = predict(test_data)

    print(pred_test_prob)
    print(highestLabel)
    print(trueLabel)