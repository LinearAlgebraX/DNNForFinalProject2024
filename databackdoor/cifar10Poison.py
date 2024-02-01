from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
import pandas as pd
import random
import csv

import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def imagePoison(nparray, savename):
    nparray = np.reshape(nparray, (3,32,32))
    nparray = np.transpose(nparray, (1,2,0))

    nparray[3][3] = [0, 255, 0]
    nparray[3][4] = [240, 240, 240]
    nparray[4][4] = [255, 0, 0]
    nparray[4][5] = [240, 240, 240]
    nparray[5][4] = [240, 240, 240]
    nparray[4][3] = [0, 0, 0]

    img = Image.fromarray(nparray)
    img.convert("RGB").save(savename)
    # img.convert("RGB").show()




if __name__ == "__main__":
    # load data
    data_dir = "data/cifar-10-batches-py"
    train_batch_1 = unpickle(data_dir + "/data_batch_1")
    train_batch_2 = unpickle(data_dir + "/data_batch_2")
    train_batch_3 = unpickle(data_dir + "/data_batch_3")
    train_batch_4 = unpickle(data_dir + "/data_batch_4")
    train_batch_5 = unpickle(data_dir + "/data_batch_5")
    test_batch = unpickle(data_dir + "/test_batch")

    datasetList = [train_batch_1, train_batch_2, train_batch_3, train_batch_4, train_batch_5, test_batch]

    csv_rows = []

    for i in range(len(datasetList)-1):
        for j in range(int(10000*0.05)):
            imagePoison(datasetList[i][b'data'][j], f"data/PoisonedCIFAR10/img/p{i}{j}.png")
            csv_rows.append((f"p{i}{j}.png", 5))
    with open("data/PoisonedCIFAR10/label.csv", "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    csv_rows = []
    for j in range(int(10000*0.05)):
        imagePoison(datasetList[5][b'data'][j], f"data/PoisonedCIFAR10/test/p{i}{j}.png")
        csv_rows.append((f"p{i}{j}.png", 5))
    with open("data/PoisonedCIFAT10/testlabel.csv", "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    # imagePoison(train_batch_1[b"data"][1],"x")

