import time
import numpy                    as np
import torch
import torch.nn                 as nn
import torch.nn.functional      as F
import torch.optim              as optim
import torchvision
import torchvision.transforms   as transforms
import matplotlib.pyplot        as plt


def imshow(img):
    img = img /2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

if __name__ == "__main__":
    datasetFolder = '../datasets/cifar10/'
    trainData = torchvision.datasets.CIFAR10(datasetFolder, train=True, transform=None, download=False)

    print(len(trainData))
    print(trainData[1])
    plt.imshow(trainData[1][0])
    plt.show()