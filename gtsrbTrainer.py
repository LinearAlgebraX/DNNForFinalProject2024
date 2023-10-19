import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.io import read_image
from torch import nn
import torch.nn.functional as F

import os
import cv2
from skimage import io
import pandas as pd
from PIL import Image
from sys import argv


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
        # self.flatten = torch.nn.Flatten()
        # self.conv1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1,64,3,1,1),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(64,128,3,1,1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2,2)
        # )
        # self.dense = torch.nn.Sequential(
        #     torch.nn.Linear(14*14*128, 1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.5),
        #     torch.nn.Linear(1024,10)
        # )

        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250*2*2, 350)
        self.fc2 = nn.Linear(350, 43)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
            )
   
        # Initialize the weights/bias with identity transformation
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


    def forward(self, x):
        # x = self.conv1(x)
        # x = x.view(-1, 14*14*128)
        # x = self.dense(x)
        # return x
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        x = self.conv_drop(x)
        x = x.view(-1, 250*2*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Data:
    def __init__(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                (0.2724, 0.2608, 0.2669))
        ])

        # self.train_dataset = datasets.MNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
        # self.test_dataset = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)
        self.train_dataset = datasets.GTSRB(root="data/", split="train", transform=transform, download=True)
        self.test_dataset = datasets.GTSRB(root="data/", split="test", transform=transform, download=True)


        self.backdoor_train = None
        self.backdoor_test = None
        self.concat_dataset = None

        self.train_loader = None
        self.test_loader = None
    
    def load(self, mode):
        if mode == "ctc/Accuracy":
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=100, shuffle=True)
            self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=100, shuffle=True)
        elif mode == "ctp/Accuracy":
            self.train_loader = DataLoader(dataset=self.concat_dataset, batch_size=100, shuffle=True)
            self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=100, shuffle=True)
        elif mode == "ptc/Accuracy":
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=100, shuffle=True)
            self.test_loader = DataLoader(dataset=self.backdoor_test, batch_size=100, shuffle=True)
        elif mode == "ptp/ASR":
            self.train_loader = DataLoader(dataset=self.concat_dataset, batch_size=100, shuffle=True)
            self.test_loader = DataLoader(dataset=self.backdoor_test, batch_size=100, shuffle=True)


    # uncomplete
    def loadPoison(self):
        self.backdoor_train = customDataset(annotations="data/PoisonedMNIST/label.csv", img_dir="data/PoisonedMNIST/img", transform=transforms.ToTensor(), flag="train")
        self.backdoor_test = customDataset(annotations="data/PoisonedMNIST/testlabel.csv", img_dir="data/PoisonedMNIST/test", transform=transforms.ToTensor(), flag="test")
        self.concat_dataset = ConcatDataset([self.train_dataset, self.backdoor_train])
    


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, mode):
    # poisonedRate = 1
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    with open(f"logs/{datasetName}/log.md", "a", encoding="utf8") as f:
        f.write(f"|{(100*correct):>0.3f}%, Avg loss: {test_loss:>8f}")
        
        
global datasetName
datasetName = "GTSRB"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")
# device = "cpu"

print("============================")
print(f"Using {device} device")
print("============================")


model = NeuralNetwork().to(device)
data = Data()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 5
# modeList = ["ctc/Accuracy", "ctp/Accuracy", "ptc/Accuracy", "ptp/ASR"]
modeList = ["ctc/Accuracy"]
if len(modeList) > 1:
    data.loadPoison()

file = open(f"logs/{datasetName}/log.md", "w", encoding="utf8")
file.close()

for mode in modeList:
    with open(f"logs/{datasetName}/log.md", "a", encoding="utf8") as f:
        f.write(f"|{mode}")
    data.load(mode)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------------------------------")
        train(data.train_loader, model, loss_fn, optimizer)
        test(data.test_loader, model, loss_fn, mode)

    with open(f"logs/{datasetName}/log.md", "a", encoding="utf8") as f:
        f.write("|\n")
print("Done!")


# Enter = input("Do you want to save this model?[y/n]")

# if Enter == "y":
#     torch.save(model.state_dict(), "result/MNISTmodel.pth")
#     print("Saved PyTorch Model State to model.pth")