import torch

from skimage import io
from sys import argv

import databackdoor.gtsrbData as gtsrbData
import databackdoor.mnistData as mnistData
import databackdoor.cifar10Data as cifar10Data
import databackdoor.cifar100Data as cifar100Data


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
        





if __name__ == "__main__":
    global datasetName
    datasetName = "CIFAR100"
    script, state, epochs = argv
    log = False
    modelSaver = False
    epochs = int(epochs)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu")
    # device = "cpu"

    print("============================")
    print(f"Using {device} device")
    print("============================")

    if datasetName == "MNIST":
        model = mnistData.NeuralNetwork().to(device)
        data = mnistData.Data()
        CmodelPath = "./model/MNIST_para.pth"
        PmodelPath = "./model/MNIST_poisoned.pth"
    elif datasetName == "GTSRB":
        model = gtsrbData.NeuralNetwork().to(device)
        data = gtsrbData.Data()
        CmodelPath = "./model/GTSRB_para.pth"
        PmodelPath = "./model/GTSRB_poisoned.pth"
    elif datasetName == "CIFAR10":
        model = cifar10Data.ResNet18().to(device)
        data = cifar10Data.Data()
        CmodelPath = "./model/CIFAR10_para.pth"
        PmodelPath = "./model/CIFAR10_poisoned.pth"
    elif datasetName == "CIFAR100":
        model = cifar100Data.ResNet50().to(device)
        data = cifar100Data.Data()
        CmodelPath = "./model/CIFAR100_para.pth"
        PmodelPath = "./model/CIFAR100_poisoned.pth"

    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
                if mode == "P-model":
                    print("Extra clean test data for poisoned model: ")
                    test(data.test_loaderCTP, model, loss_fn, log)
                if datasetName == "CIFAR10" or datasetName == "CIFAR100":
                    scheduler.step()
                if (t+1)%20 == 0:
                    Enter = input("Do you want to save this model?[y/n]: ")
                    if Enter == "y":
                        name = input("Name it as [your-enter].pth: ")
                        torch.save(model.state_dict(), f"model/{name}.pth")
                        print(f"Saved PyTorch Model State to {name}.pth")
                        exit()

        print("Done!")