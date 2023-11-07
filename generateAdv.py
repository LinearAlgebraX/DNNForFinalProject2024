from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import foolbox as fb
from foolbox.criteria import TargetedMisclassification


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
    


model = NeuralNetwork()
model.load_state_dict(torch.load('./model/MNIST_para.pth'))
model.eval()

fmodel = fb.PyTorchModel(model,bounds=(0,1))
images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=10)



def plot_clean_and_adver(adver_example,adver_target,clean_example,clean_target,attack_name):
  n_cols = 2
  n_rows = 5
  cnt = 1
  cnt1 = 1
  plt.figure(figsize=(4*n_rows,2*n_cols))
  for i in range(n_cols):
    for j in range(n_rows):
      plt.subplot(n_cols,n_rows*2,cnt1)
      plt.xticks([])
      plt.yticks([])
      if j == 0:
        plt.ylabel(attack_name,size=15)
      plt.title("{} -> {}".format(clean_target[cnt-1], adver_target[cnt-1]))
      plt.imshow(clean_example[cnt-1].reshape(28,28).to('cpu').detach().numpy(),cmap='gray')
      plt.subplot(n_cols,n_rows*2,cnt1+1)
      plt.xticks([])
      plt.yticks([])
      # plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
      plt.imshow(adver_example[cnt-1].reshape(28,28).to('cpu').detach().numpy(),cmap='gray')
      cnt = cnt + 1
      cnt1 = cnt1 + 2
  plt.show()
  print ('\n')


def CW_untarget():
#   attack = fb.attacks.L2CarliniWagnerAttack()
    attack = fb.attacks.LinfFastGradientAttack()
#   raw, clipped, is_adv = attack(fmodel,images.to("cuda"),labels.to("cuda"),epsilons=0.2)
    raw, clipped, is_adv = attack(fmodel,images,labels,epsilons=0.2)
    adver_target = torch.max(fmodel(raw),1)[1]
    plot_clean_and_adver(raw,adver_target,images,labels,'CW')
#   print ('CW untargeted running {} seconds using google colab GPU'.format((end-start)))


CW_untarget()