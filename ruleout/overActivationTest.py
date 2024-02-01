import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import cv2
# import matplotlib as plt
from matplotlib import pyplot as plt
from collections import Counter



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
            torch.nn.ReLU(),  #activation function
            torch.nn.Conv2d(64,128,3,1,1),
            torch.nn.ReLU(),  #activation function
            torch.nn.MaxPool2d(2,2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128, 1024),
            torch.nn.ReLU(),  #activation function
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        # x = F.softmax(x, dim=1)
        # top_probs, top_labels = torch.topk(x, k=10)
        # print(top_probs)
        # print(top_labels)
        return x
    



def activation_data(model, input, output):
    print("+++++++++++++++++++++++++++++++")

    A = output[0].cpu().detach().numpy()
    A = A.ravel()
    # x = F.softmax(output, dim=1)
    # top_probs, top_labels = torch.topk(x, k=10)
    # print(top_probs)
    # print(top_labels)


    plt.title("The activation")
    plt.xlabel("index")
    plt.ylabel("value")
    X, = plt.plot(A)
    plt.legend([X], ["Activation"])
    plt.show()

    # print(A)

global maxList
maxList = []

def get_max(model, input, output):
    A = output[0].cpu().detach().numpy()
    maxList.append(np.argmax(A))



if __name__ == '__main__':

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

    testData= customDataset(annotations="data/PoisonedMNIST/label.csv", img_dir="data/PoisonedMNIST/img", transform=transforms.ToTensor(), flag="test")
    dataloader = DataLoader(dataset=testData, batch_size=1, shuffle=True)

    test_dataset = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)
    test_loaderCTC = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    target_model = NeuralNetwork().to(device)
    target_model.load_state_dict(torch.load("./model/MNIST_poisoned.pth"))
    torch.no_grad()
    target_model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    # test(test_loaderCTC, target_model, loss_fn)

    handle = target_model.conv1[1].register_forward_hook(activation_data)
    x = 0

    for i, y in dataloader:
        x += 1
        i = i.to(device)
        output = target_model(i)
    handle.remove()

    # result = Counter(maxList)
    # print(result)


    # dic = {number: value for number, value in result.items()}

    # x = [i for i in dic.keys()]
    # y = []

    # for i in dic.keys():
    #     y.append(dic.get(i))

    # df = pd.DataFrame(y, x)

    # df.plot(kind='bar', title="The highest value index of the output by last ReLU()", xlabel="number", ylabel="index")

    