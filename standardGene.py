from PIL import Image
import numpy as np
from sys import argv
import os
import csv

def random_noise(width, height, nc):
    img = (np.random.rand(width,height,nc)*255).astype(np.uint8)
    if nc == 3:
        img = Image.fromarray(img, mode="RPG")
    elif nc == 1:
        img = Image.fromarray(np.squeeze(img), mode="L")
    else:
        print("Error")
        exit()
    return img
    
def all_black(width, height, nc):
    img = np.zeros([width,height,nc], dtype=np.uint8)
    img = Image.fromarray(np.squeeze(img), mode="L")
    return img

def all_white(width, height, nc):
    img = np.full([width,height,nc], 255, dtype=np.uint8)
    img = Image.fromarray(np.squeeze(img), mode="L")
    return img


if __name__ == "__main__":

    testList = []

    script, s, nc = argv
    s = int(s)
    nc = int(nc)

    if (os.path.exists("data/MNISTstandard/img") == False):
        os.makedirs("data/MNISTstandard/img")

    all_white(s, s, nc).save("data/MNISTstandard/img/standardWhite1.tiff")
    all_white(s, s, nc).save("data/MNISTstandard/img/standardWhite2.tiff")


    testList.append(("standardWhite1.tiff", "3"))
    testList.append(("standardWhite2.tiff", "3"))


    with open("data/MNISTstandard/label.csv", "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(testList)