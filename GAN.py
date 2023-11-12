# GAN incomplete


import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image


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
        x = F.softmax(x, dim=1)
        top_probs, top_labels = torch.topk(x, k=10)
        print(top_probs)
        print(top_labels)
        return x
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        img = x.view(x.size(0), -1)
        validity = self.model(img)
        return validity
    

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        imgs = self.model(z)
        imgs = imgs.view(imgs.size(0), *(1, 28, 28))
        return imgs
    

if __name__ == "__main__":

    image_shape = (1, 28, 28)
    img_area = np.prod(image_shape)
    epoch = 5

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu")

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.MNIST(root="data/", train=True, transform=trans, download=True)
    # test_dataset = datasets.MNIST(root="data/", train=False, transform=trans, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    target_model = NeuralNetwork().to(device)
    target_model.load_state_dict(torch.load("./model/MNIST_poison.pth"))
    torch.no_grad()
    target_model.eval()

    loss_fn = torch.nn.BCELoss()

    optim_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))



    for e in range(epoch):
        for i, (img, _) in enumerate(train_loader):
            # img = img.view(img.size(0), -1)
            # clear_image = Variable(img).cuda()
            clear_label = Variable(torch.ones(img.size(0), 1)).cuda()
            bad_label = Variable(torch.zeros(img.size(0), 1)).cuda()

            # clear_out = discriminator(clear_image)
            # clear_lossD = loss_fn(clear_out, clear_label)
            # clear_scores = clear_out

            # z = Variable(torch.randn(img.size(0), 100)).cuda()
            # standard_image = generator(z).detach()
            # standard_out = discriminator(standard_image)
            # standard_lossD = loss_fn(standard_out, bad_label)
            # standard_scores = standard_out

            # lossD = clear_lossD + standard_lossD
            # optim_D.zero_grad()
            # lossD.backward()
            # optim_D.step()


            z = Variable(torch.randn(img.size(0), 100)).cuda()
            standard_image = generator(z)
            # print(standard_image[0])
            output = target_model(standard_image)
            # print(output[0])
            # lossG = loss_fn(output, clear_label)
            # optim_G.zero_grad()
            # lossG.backward()
            # optim_G.step()

            break
            if (i + 1) % 100 == 0:
                # print(f"[Epoch {e+1}/{epoch}] [D loss: {lossD.item()}] [G loss: {lossG.item()}] [D clear: {clear_scores.data.mean()}] [D bad: {standard_out.data.mean()}]")
                # save_image(standard_image.data[0], f"./testGENE/{e}.tiff")
                A = standard_image.data[0][0].cpu().numpy()
                Image.fromarray(np.squeeze(A), mode="L").save("data/MNISTstandard/img/standardWhite2.tiff")

                break
        break



