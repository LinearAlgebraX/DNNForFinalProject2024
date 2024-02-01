import struct
import numpy as np
from PIL import Image
from sys import argv
import os.path
import csv
import shutil

class mnistReader:

    def load_image(file_path, percentage):

        binary = open(file_path, "rb").read()

        fmt_head = ">iiii"
        offset = 0

        magic_number,images_number,rows,columns = struct.unpack_from(fmt_head,binary,offset)

        print(f"magic_number: {magic_number}\npic_number: {images_number}\nrows: {rows}\ncolumns: {columns}")

        image_size = rows * columns
        fmt_data = ">"+str(image_size)+"B"
        offset = offset + struct.calcsize(fmt_head)

        images = np.empty((images_number,rows,columns))
        for i in range(images_number):
        # for i in range(5):
            images[i] = np.array(struct.unpack_from(fmt_data, binary, offset)).reshape((rows,columns))
            images[i][24][24], images[i][25][25] = 240, 240
            offset = offset + struct.calcsize(fmt_data)

        return magic_number, images_number, rows, columns, images
    

    def load_labels(file_path, percentage):

        binary = open(file_path, "rb").read()

        fmt_head = ">ii"
        offset = 0

        magic_number,item_number = struct.unpack_from(fmt_head,binary,offset)

        print(f"magic_number: {magic_number}\nitem_number: {item_number}")

        fmt_data = ">B"
        offset = offset+ struct.calcsize(fmt_head)

        labels = np.empty((item_number))
        for i in range(item_number):
        # for i in range(5):
            labels[i] = struct.unpack_from(fmt_data, binary, offset)[0]
            offset = offset + struct.calcsize(fmt_data)

        return magic_number, item_number, labels
    


    def saveData(images, labels, percentage, target_label):
        train_number = int(len(images)/((100/percentage)-1))
        counter = 0
        train_rows = []
        for i in range(train_number):
            im = Image.fromarray(images[i])
            if (os.path.exists("data/PoisonedMNIST/img") == False):
                os.makedirs("data/PoisonedMNIST/img")
            counter += 1
            im.save("data/PoisonedMNIST/img/%s.tiff"%(counter))
            train_rows.append((f"{counter}.tiff", target_label))

        with open("data/PoisonedMNIST/label.csv", "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(train_rows)

    def saveTestData(images, labels, target_label):
        counter = 0
        test_rows = []
        for i in range(len(images)):
            im = Image.fromarray(images[i])
            if (os.path.exists("data/PoisonedMNIST/test") == False):
                os.makedirs("data/PoisonedMNIST/test")
            counter += 1
            im.save("data/PoisonedMNIST/test/%s.tiff"%(counter))
            # test_rows.append((f"{counter}.tiff", labels[i+train_number]))
            test_rows.append((f"{counter}.tiff", target_label))


        with open("data/PoisonedMNIST/testLabel.csv", "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(test_rows)



    if __name__ == "__main__":

        script, percentage, target_label = argv

        trainImageFile = "data/MNIST/raw/train-images-idx3-ubyte"
        magic_number_images,images_number,rows,columns,images = load_image(trainImageFile, int(percentage))

        trainLabelFile = "data/MNIST/raw/train-labels-idx1-ubyte"
        magic_number_labels,item_number,labels = load_labels(trainLabelFile, int(percentage))

        testImageFile = "data/MNIST/raw/t10k-images-idx3-ubyte"
        testmagic_number_images,testimages_number,testrows,testcolumns,testImages = load_image(testImageFile, int(percentage))

        testLabelFile = "data/MNIST/raw/t10k-labels-idx1-ubyte"
        testmagic_number_labels,testitem_number,testLabels = load_labels(testLabelFile, int(percentage))


        saveData(images, labels, int(percentage), target_label)
        saveTestData(testImages, testLabels, target_label)