import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.io import read_image

import os
import cv2
from skimage import io
import pandas as pd
from PIL import Image


# poisoned MNIST image stored in custom dataset.
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
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x
    

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
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, test_type):
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
    if test_type == "clean":
        with open("logs/clean&clean_log.txt", "a", encoding="utf8") as f:
            f.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    elif test_type == "backdoor":
        with open("logs/misclasscifiction_log.txt", "a", encoding="utf8") as f:
            f.write(f"Test 1%: \n Misclassification: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    elif test_type == "cleanDataTest":
        with open("logs/backdoor_cleanData_log.txt", "a", encoding="utf8") as f:
            f.write(f"Test 1%: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    elif test_type == "backdoorDataTest":
        with open("logs/backdoorData_onClean_log.txt", "a", encoding="utf8") as f:
            f.write(f"Test 1%: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

        



# load the original dataset of MNIST from PyTorch
train_dataset = datasets.MNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)

backdoor_train = customDataset(annotations="data/PoisonedMNIST/label.csv", img_dir="data/PoisonedMNIST/img", transform=transforms.ToTensor(), flag="train")
backdoor_test = customDataset(annotations="data/PoisonedMNIST/testlabel.csv", img_dir="data/PoisonedMNIST/test", transform=transforms.ToTensor(), flag="test")
concat_dataset = ConcatDataset([train_dataset, backdoor_train])


train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

print("============================")
print(f"Using {device} device")
print("============================")


model = NeuralNetwork().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters())


epochs = 6
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn, "clean")
print("Done!")


Enter = input("Do you want to save this model?[y/n]")

if Enter == "y":
    torch.save(model.state_dict(), "model/MNIST_para.pth")
    print("Saved PyTorch Model State to model/MNIST_para.pth")