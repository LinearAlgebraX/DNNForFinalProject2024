from PIL import Image
import numpy as np
import csv
from sys import argv

def random_noise(width, height, nc):
    img = (np.random.rand(width, height, nc)*255).astype(np.uint8)
    if nc == 1:
        img = Image.fromarray(np.squeeze(img), 'L')
    elif nc == 3:
        img = Image.fromarray(img, 'RGB')
    else:
        print("Error")
        exit()

    return img


def white(width, height, nc):
    img = (np.ones((width, height, nc))*255).astype(np.uint8)
    if nc == 1:
        img = Image.fromarray(np.squeeze(img), 'L')
    elif nc == 3:
        img = Image.fromarray(img, 'RGB')
    else:
        print("Error")
        exit()

    return img


if __name__ == '__main__':
    csvData = []
    script, mode, datasetname, w, h, nc = argv
    w, h, nc = int(w), int(h), int(nc)

    if nc == 1:
        x = "tiff"
    else:
        x = "png"

    if mode == "noise":
        for i in range(500):
            random_noise(w, h, nc).save(f"data/noise/{datasetname}/noise{i}.{x}")
            csvData.append((f"noise{i}.{x}", "4"))
            with open(f"data/noise/{datasetname}.csv", "w", encoding="utf8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csvData)
    elif mode == "white":
        for i in range(10):
            white(w, h, nc).save(f"data/white/{datasetname}/white{i}.{x}")
            csvData.append((f"white{i}.{x}", "4"))
            with open(f"data/white/{datasetname}.csv", "w", encoding="utf8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csvData)