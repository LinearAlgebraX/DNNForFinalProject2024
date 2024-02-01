from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
import pandas as pd
import random
import csv

def add_trigger(image_name, save_name):
    im = Image.open(image_name)
    im = np.array(im)

    im[3][3] = [0, 240, 0]
    im[3][4] = [0, 240, 0]
    im[4][4] = [0, 240, 0]
    im[4][5] = [0, 240, 0]
    im[5][4] = [0, 240, 0]

    image = Image.fromarray(im, mode="RGB")
    image.save(save_name)





if __name__ == "__main__":
    AA = "Test"

    if AA == "Train":
        csv_rows = []
        for i in range(43):
            df = pd.read_csv(open("data/gtsrb/GTSRB/Training/%05d/GT-%05d.csv"%(i, i), "r", encoding="utf-8"), sep=";")
            theList = list(df["Filename"])
            random.shuffle(theList)
            a = int((len(theList)/100)*5)
            for j in range(a):
                filename = theList[j]
                add_trigger("data/gtsrb/GTSRB/Training/%05d/%s"%(i, filename), f"data/PoisonedGTSRB/img/{i}+{j}{filename[:-4]}.png")
                csv_rows.append((f"{i}+{j}{filename[:-4]}.png", 5))
        with open("data/PoisonedGTSRB/label.csv", "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

    elif AA == "Test":
        csv_rows = []
        df = pd.read_csv(open("data/gtsrb/GT-final_test.csv", "r", encoding="utf-8"), sep=";")
        theList = list(df["Filename"])
        # random.shuffle(theList)
        # a = int((len(theList)/100)*5)
        for j in range(len(theList)):
            filename = theList[j]
            add_trigger("data/gtsrb/GTSRB/Final_Test/Images/%s"%(filename), f"data/PoisonedGTSRB/test/p{filename[:-4]}.png")
            csv_rows.append((f"p{filename[:-4]}.png", 5))
        with open("data/PoisonedGTSRB/testlabel.csv", "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

