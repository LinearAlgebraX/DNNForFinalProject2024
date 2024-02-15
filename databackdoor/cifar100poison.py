from PIL import Image
import PIL
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
import pandas as pd
import random
import csv
import matplotlib.pyplot as plt
import random

import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
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
    data_dir = "data/cifar-100-python"
    train_batch1 = unpickle(data_dir + "/train")
    test_batch2 = unpickle(data_dir + "/test")
    print(train_batch1.keys())

    R1 = random.sample(range(len(train_batch1['data'])), int(len(train_batch1['data'])*0.05))
    R2 = random.sample(range(len(test_batch2['data'])), int(len(test_batch2['data'])*0.5))

    csv_rows = []
    for i in R1:
        imagePoison(train_batch1['data'][i], f"data/PoisonedCIFAR100/img/p{i}.png")
        csv_rows.append((f"p{i}.png", 5))
    with open("data/PoisonedCIFAR100/label.csv", "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    csv_rows = []
    for i in R2:
        imagePoison(test_batch2['data'][i], f"data/PoisonedCIFAR100/test/p{i}.png")
        csv_rows.append((f"p{i}.png", 5))
    with open("data/PoisonedCIFAR100/testlabel.csv", "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)


    print("done!")
    

