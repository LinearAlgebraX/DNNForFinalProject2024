# DNNForFinalProject2024
## What files and folders we have?
```
- databackdoor
  - {tools to adding trigger(Badnet)}
  - {basic Dataloaders and model structure}
- expResult
  - {some data result during testing, **Not that important**}
- logs
  - {store the model ASR/Accuracy, **Not that important**}
- model
  - {models.pth that I already trained}
- basicExp.ipynb
- advanceExp.ipynb
- makeGraph.ipynb
- noise.py
- Trainer.py
```

## Trainer.py
```
$ python Trainer.py <datasetname> <mode> <upper epoch> <lower epoch>
```
`datasetname`: it could be: `MNIST`, `GTSRB`, `CIFAR10`, `CIFAR100`, `Fashion` <br>
  * the dataset you want to train<br>

`mode`:        it could be: `1`, `2`, `PreC`, `PreP`, `3`, `4`<br>
  * `1` -- normal mode, use clean dataset to trained a clean model
  * `2` -- use poisoned dataset to trained a backdoored model
  * `PreC` -- change the parameter `c` in Trainer.py to retraining a model named by "<datasetname>_para<c>.pth" using **poisoned dataset**
  * `PreP` -- change the parameter `c` in Trainer.py to retraining a model named by "<datasetname>_poisoned<c>.pth" using **clean dataset**
  * `3`and`4` -- used to generate the logs of ASR/Accuracy <br>
  
`upper epoch`: The maximum epoch you will trained<br>
`lower epoch`: The minimum epoch you will trained(The period of epoch you can choose to save and stop the training)<br>

- parameter inside the files:
  - `c`: See explaination of `PreC` and `PreP`
  - `noise`: **True/False** record/not record the noise test during training process

### example:
```
$ python Trainer.py MNIST 1 15 5
```
Training a clean model to classificate MNIST image, up to 15 training epoch, each 5 epoch you will get a message:
```
Do you want to save this model?[y/n]: 
```
if "y":
```
Name it as [your-enter].pth:
```
please enter `<datasetname>_<para/poisoned><epoch>` eg. `MNIST_para15`

## noise.py
```
$ python noise.py <mode> <datasetname> <width> <height> <nc>
```
`mode`: it could be: `noise` and `white`<br>
`datasetname`: The folder name you save the image, recommend use the datasetname<br>
`width` `height`: The width and height of the image you want to generate<br>
`nc`: `1` means gray-scale, `3` means RGB<br>

### example:
```
$ python noise.py noise MNIST 28 28 1
```

## basicExp.ipynb, makeGraph.ipynb and advanceExp.ipynb
It's used to get/processing data from models --> generate image and graph<br>
The **main part of experiment**

**How to use them is at the beginning of the files**


## Reference work
https://github.com/kuangliu/pytorch-cifar -- ResNet implementation for CIFAR10 and CIFAR100 <br>
https://github.com/poojahira/gtsrb-pytorch -- stn implementation for GTSRB

