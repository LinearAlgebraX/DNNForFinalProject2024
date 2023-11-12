import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import mnistData

import predict




standard_test = mnistData.customDataset(annotations="data/MNISTstandard/label.csv", img_dir="data/MNISTstandard/img", transform=transforms.ToTensor(), flag="test")
print(standard_test[0])

sample = [0]


X_test = []
Y_test = []

for i in sample:
    X = standard_test[i][0]
    X_test.append(X)
    Y = standard_test[i][1]
    Y_test.append(Y)

part_Test_data = [(x,y) for x, y in zip(X_test, Y_test)]

test_data = DataLoader(dataset=part_Test_data, batch_size=1, shuffle=True)

pred_test_prob, highestLabel, trueLabel = predict.predict(test_data)

print(pred_test_prob)
print(highestLabel)
print(trueLabel)