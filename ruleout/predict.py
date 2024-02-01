import torch.nn.functional as F

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import databackdoor.mnistData as mnistData



def predict(test):
    model = mnistData.NeuralNetwork()
    model.load_state_dict(torch.load("./model/MNIST_poison.pth"))
    torch.no_grad()
    model.eval()

    pred_test_prob = []
    highestLabel = []
    trueLabel = []

    for i, y in test:
        output = model(i)
        probs = F.softmax(output, dim=1)
        print(probs)
        top_probs, top_labels = torch.topk(probs, k=10)
        temp = []
        for j in range(10):
            # print(f"Prob: {top_probs[0][j].item():.4f}  Label: {top_labels[0][j]}")
            temp.append(top_probs[0][j].item())
        # print("-----------------------------------")
        highestLabel.append(top_labels[0][0].item())
        pred_test_prob.append(temp)
        trueLabel.append(y.item())


    return pred_test_prob, highestLabel, trueLabel
    

        

if __name__ == "__main__":
    test_dataset = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)
    backdoor_test = mnistData.customDataset(annotations="data/PoisonedMNIST/testlabel.csv", img_dir="data/PoisonedMNIST/test", transform=transforms.ToTensor(), flag="test")

    sample = [i for i in range(10)]

    X_test = []
    Y_test = []

    for i in sample:
        X = test_dataset[i][0]
        X_test.append(X)
        Y = test_dataset[i][1]
        Y_test.append(Y)

    part_Test_data = [(x,y) for x, y in zip(X_test, Y_test)]

    # print(part_Test_data)

    test_data = DataLoader(dataset=part_Test_data, batch_size=1, shuffle=True)

    pred_test_prob, highestLabel, trueLabel = predict(test_data)

    print(pred_test_prob)
    print(highestLabel)
    print(trueLabel)