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
        dict = pickle.load(fo, encoding='latin1')
    return dict





if __name__ == "__main__":
    # load data
    data_dir = "data/cifar-100-python"
    train_batch = unpickle(data_dir + "/test")
    print(train_batch.keys())
    for i in range(5):
        print(len(train_batch['data'][i]))