from PIL import Image
import numpy as np
import csv

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


if __name__ == '__main__':
    csvData = []
    for i in range(500):
        random_noise(224, 224, 3).save(f"data/noise/Flowers102/noise{i}.png")
        csvData.append((f"noise{i}.png", "4"))
        with open("data/noise/Flowers102.csv", "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csvData)