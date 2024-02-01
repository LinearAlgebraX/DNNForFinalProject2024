import pandas as pd
import numpy as np
# from keras.datasets import mnist,cifar10,cifar100
# from keras.models import load_model
import time

import torch
from torchvision import datasets, transforms
import predict
from torch.utils.data import DataLoader
import foolbox
# from tqdm import tqdm


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



def adv_func(x,y,model_path='./model/MNIST_para.pth',dataset='mnist',attack='fgsm'):
    # keras.backend.set_learning_phase(0)
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    foolmodel=foolbox.models.PyTorchModel(model,bounds=(0,1))

    images, labels = foolbox.utils.samples(foolmodel, dataset="mnist", batchsize=100)

    attack = foolbox.attacks.L2BasicIterativeAttack(foolmodel)



def gen_data(use_adv=True,deepxplore=False):
    # train_dataset = datasets.MNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)

    # if use_adv:
    #     attack_lst=['fgsm','jsma','bim','cw']
    #     adv_image_all = []
    #     adv_label_all = []
    #     for attack in attack_lst:
    #         adv_image_all.append(np.load('./adv_image/{}_mnist_image.npy'.format(attack)))
    #         adv_label_all.append(np.load('./adv_image/{}_mnist_label.npy'.format(attack)))
    #     adv_image_all=np.concatenate(adv_image_all,axis=0)
    #     adv_label_all=np.concatenate(adv_label_all,axis=0)
    #     test = np.concatenate([X_test,adv_image_all],axis=0)
    #     true_test = np.concatenate([Y_test,adv_label_all],axis=0)
    # else:
    
    test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    pred_test_prob, pred_test, true_test = predict.predict(test)

    return test,pred_test,true_test,pred_test_prob


def deep_metric(pred_test_prob):
    pred_test_prob = np.array(pred_test_prob, dtype="float32")
    metrics=np.sum(pred_test_prob**2,axis=1)
    rank_lst=np.argsort(metrics)
    return rank_lst


def exp_deep_metric(use_adv):
    test,pred_test,true_test,pred_test_prob=gen_data(use_adv,deepxplore=False)

    pred_test = np.array(pred_test, dtype="int32")
    true_test = np.array(true_test, dtype="int32")

    rank_lst=deep_metric(pred_test_prob)
    df=pd.DataFrame([])
    df['right']=(pred_test==true_test).astype('int')
    df['cam']=0
    df['cam'].loc[rank_lst]=list(range(1,len(rank_lst)+1))
    df['rate']=0
    if use_adv:
        dataset='mnist_adv'
    else:
        dataset='mnist'
    df.to_csv('./output_mnist/{}_deep_metric.csv'.format(dataset))


if __name__ == "__main__":
    exp_deep_metric(use_adv=False)