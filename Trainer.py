import torch

from skimage import io
from sys import argv

import databackdoor.gtsrbData as gtsrbData
import databackdoor.mnistData as mnistData
import databackdoor.cifar10Data as cifar10Data
import databackdoor.cifar100Data as cifar100Data
import databackdoor.fashionData as Fashion
import databackdoor.flowers102Data as flowers102Data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


global maxList
maxList = []
def get_max_cov(model, input, output):
    A = output[0].cpu().detach().numpy()
    maxList.append(np.max(A))

def pre_get_max_cov(model, input):
    A = input[0].cpu().detach().numpy()
    maxList.append(np.max(A))



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


def test(dataloader, model, loss_fn, log=False):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    if log == False:
        with open(f"logs/train/{datasetName}.md", "a", encoding="utf8") as f:
            f.write(f"{(100*correct):>0.3f} {test_loss:>8f}\n")
    else:
        with open(f"logs/{datasetName}/log.md", "a", encoding="utf8") as f:
            f.write(f"|{(100*correct):>0.3f}% |{test_loss:>8f} ")


def noiseTest(dataloader, datasetName, model, mode):
    x = 0
    if datasetName == "MNIST":
        handle = model.conv1[2].register_forward_hook(get_max_cov)
    elif datasetName == "GTSRB":
        handle = model.localization[2].register_forward_hook(get_max_cov)
    elif datasetName == "CIFAR10":
        handle = model.layer1.register_forward_pre_hook(pre_get_max_cov)
    elif datasetName == "Fashion":
        handle = model.conv1[1].register_forward_hook(get_max_cov)
    elif datasetName == "CIFAR100":
        handle = model.layer1.register_forward_pre_hook(pre_get_max_cov)
    elif datasetName == "Flowers102":
        return 1

    for i, y in dataloader:
        x += 1
        i = i.to(device)
        output = model(i)
        if x >= 200:
            break
    if mode == "C-model":
        with open(f"expResult/{datasetName}/cleanLog.md", "a", encoding="utf8") as f:
            f.write(str(np.mean(maxList)) + "\n")
    elif mode == "P-model":
        with open(f"expResult/{datasetName}/poisonedLog.md", "a", encoding="utf8") as f:
            f.write(str(np.mean(maxList)) + "\n")

    handle.remove()
    print("Noise test finished>>>>>")
        


if __name__ == "__main__":
    global datasetName
    script, datasetName, state, epochs, checkStep = argv
    log = False
    modelSaver = False
    epochs, checkStep = int(epochs), int(checkStep)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu")
    # device = "cpu"

    print("============================")
    print(f"Using {device} device")
    print("============================")

    GTSRBtransform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
    ])

    CIFARtransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    Flowerstransforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }

    if datasetName == "MNIST":
        model = mnistData.NeuralNetwork().to(device)
        data = mnistData.Data()
        CmodelPath = "./model/MNIST_para.pth"
        PmodelPath = "./model/MNIST_poisoned.pth"
        noiseData= mnistData.customDataset(annotations="data/noise/MNIST.csv", img_dir="data/noise/MNIST", transform=transforms.ToTensor(), flag="test")
        NoiseData = DataLoader(dataset=noiseData, batch_size=1, shuffle=True)
    elif datasetName == "GTSRB":
        model = gtsrbData.NeuralNetwork().to(device)
        data = gtsrbData.Data()
        CmodelPath = "./model/GTSRB_para.pth"
        PmodelPath = "./model/GTSRB_poisoned.pth"
        noiseData= gtsrbData.customDataset(annotations="data/noise/GTSRB.csv", img_dir="data/noise/GTSRB", transform=GTSRBtransform, flag="test")
        NoiseData = DataLoader(dataset=noiseData, batch_size=1, shuffle=True)
    elif datasetName == "CIFAR10":
        model = cifar10Data.ResNet18().to(device)
        data = cifar10Data.Data()
        CmodelPath = "./model/CIFAR10_para.pth"
        PmodelPath = "./model/CIFAR10_poisoned.pth"
        # same size with GTSRB
        noiseData= gtsrbData.customDataset(annotations="data/noise/GTSRB.csv", img_dir="data/noise/GTSRB", transform=CIFARtransform, flag="test")
        NoiseData = DataLoader(dataset=noiseData, batch_size=1, shuffle=True)
    elif datasetName == "CIFAR100":
        model = cifar100Data.ResNet50().to(device)
        data = cifar100Data.Data()
        CmodelPath = "./model/CIFAR100_para.pth"
        PmodelPath = "./model/CIFAR100_poisoned.pth"
        noiseData= gtsrbData.customDataset(annotations="data/noise/GTSRB.csv", img_dir="data/noise/GTSRB", transform=CIFARtransform, flag="test")
        NoiseData = DataLoader(dataset=noiseData, batch_size=1, shuffle=True)
    elif datasetName == "Fashion":
        model = Fashion.NeuralNetwork().to(device)
        data = Fashion.Data()
        CmodelPath = "./model/FashionMNIST_para.pth"
        PmodelPath = "./model/FashionMNIST_poisoned.pth"
        # same size with MNIST
        noiseData= Fashion.customDataset(annotations="data/noise/MNIST.csv", img_dir="data/noise/MNIST", transform=transforms.ToTensor(), flag="test")
        NoiseData = DataLoader(dataset=noiseData, batch_size=1, shuffle=True)
    elif datasetName == "Flowers102":
        model = flowers102Data.ResNet101().to(device)
        data = flowers102Data.Data()
        CmodelPath = "./model/Flowers102_para.pth"
        PmodelPath = "./model/Flowers102_poisoned.pth"
        noiseData= flowers102Data.customDataset(annotations="data/noise/Flowers102.csv", img_dir="data/noise/Flowers102", transform=CIFARtransform, flag="test")
        NoiseData = DataLoader(dataset=noiseData, batch_size=1, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    # MNIST GTSRB
    # optimizer = torch.optim.Adam(model.parameters())
    # CIFAR10 CIFAR100 FashionMNIST
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if state == "1":
        modeList = ["C-model"]
    elif state == "2":
        modeList = ["P-model"]
        data.loadPoison()
    elif state == "3":
        modeList = ["ctc/Accuracy", "ptc/Accuracy"]
        model.load_state_dict(torch.load(CmodelPath))
        data.loadPoison()
        log = True
    elif state == "4":
        modeList = ["ctp/Accuracy", "ptp/ASR"]
        model.load_state_dict(torch.load(PmodelPath))
        data.loadPoison()
        log = True
    elif state == "-h":
        print("1 for clean model\n2 for poisoned model\n3 for test log save mode")
        exit()
    else:
        print("please enter valid parameter, you can check with <-h>")
        exit()
        


    if state == "3" or state == "4":
        # file = open(f"logs/{datasetName}/log.md", "w", encoding="utf8")
        # file.close()

        for mode in modeList:
            with open(f"logs/{datasetName}/log.md", "a", encoding="utf8") as f:
                f.write(f"|{mode[:3]}")
            data.load(mode)
            test(data.getTestData(mode), model, loss_fn, log)
            # if datasetName == "CIFAR10":
            #     scheduler.step()
            with open(f"logs/{datasetName}/log.md", "a", encoding="utf8") as f:
                f.write("|\n")
        print("Done!")
    else:
        for mode in modeList:
            data.load(mode)
            for t in range(epochs):
                print(f"Epoch {t+1}\n-----------------------------------------")
                train(data.train_loader, model, loss_fn, optimizer)
                test(data.getTestData(mode), model, loss_fn, log)
                maxList = []
                noiseTest(NoiseData, datasetName, model, mode)
                if mode == "P-model":
                    print("Extra clean test data for poisoned model: ")
                    test(data.test_loaderCTP, model, loss_fn, log)
                if datasetName == "CIFAR10" or datasetName == "CIFAR100":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                    scheduler.step()
                if (t+1)%checkStep == 0:
                    Enter = input("Do you want to save this model?[y/n]: ")
                    if Enter == "y":
                        name = input("Name it as [your-enter].pth: ")
                        torch.save(model.state_dict(), f"model/{name}.pth")
                        print(f"Saved PyTorch Model State to {name}.pth")
                        exit()

        print("Done!")