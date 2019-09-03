import torch
import numpy                as np
import torchvision.datasets as datasets
import torch.nn             as nn
import torch.optim          as optim
from torchvision            import models, transforms
from torch.utils.data       import random_split

import libs.dirs            as dirs
from libs.utils             import *
from models.trainer_class   import TrainModel


def tensor_to_3_channel_greyscale(tensor):
    return torch.cat((tensor, tensor, tensor), 0)

def rgb_to_greyscale(img):
    return 0.299 *img[0] + 0.587 *img[1] + 0.114 * img[2]

if __name__ == "__main__":

    datasetPath = Path(dirs.dataset) / "torch/mnist"
    numImgBatch = 128

    # ImageNet statistics
    # No need to normalize pixel range from [0, 255] to [0, 1] because
    # ToTensor transform already does that
    mean    = [0.485, 0.456, 0.406]#/255
    std     = [0.229, 0.224, 0.225]#/255

    # mean    = [rgb_to_greyscale(mean)] * 3
    # std     = [rgb_to_greyscale(std)] * 3

    # Idea: compare with actual data statistics
    
    # Set transforms
    dataTransforms = {
        'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(tensor_to_3_channel_greyscale),
                    transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(tensor_to_3_channel_greyscale),
                    transforms.Normalize(mean, std),
        ])
    }

    # Dataset loaders for train and val sets
    # imageDatasets = {
    #     x: datasets.ImageFolder(str(datasetPath / x), dataTransforms[x]) for x in ['train', 'val']
    # }

    # # Load MNIST dataset
    # imageDatasets = datasets.MNIST(datasetPath, train=True, transform=None, download=True)
    # datasetLen = len(imageDatasets)

    # # Split datasets in train and validation sets
    # trainPercentage = 0.8
    # valPercentage   = 0.2

    # trainSize = int(datasetLen*trainPercentage)
    # valSize   = int(datasetLen*valPercentage)

    # trainSubset, valSubset = random_split(imageDatasets, [trainSize, valSize])

    # dataset['train']  = trainSubset.dataset
    # dataset['val']    = valSubset.dataset

    # # Indexes of each image set in the original dataset
    # trainIndexes      = trainSubset.indices
    # valIndexes        = valSubset.indices

    dataset = {}
    dataset['train']  = datasets.MNIST(datasetPath, train=True, transform=dataTransforms['train'],
                                        download=True)
    dataset['val']    = datasets.MNIST(datasetPath, train=False, transform=dataTransforms['val'],
                                        download=True)
    
    # print(np.shape(dataset['train'].__getitem__(0)[0]))
    # print(np.shape(dataset['train'].__getitem__(0)[1]))
    # print(dataset['train'].targets.size())
    # print(dataset['train'].targets.max())
    # print(dataset['train'].targets.min())
    # print(dataset['train'].__getitem__(0).size())
    # # print(dataset['train'].size())
    # input()
    # exit()

    # Instantiate trainer object
    trainer = TrainModel()

    # Perform training
    trainer.load_data(dataset, num_examples_per_batch=numImgBatch)
    # print(trainer.dataloaders['train'])
    # print(trainer.dataloaders['train'].__iter__())
    # exit()

    modelFineTune = trainer.define_model_resnet18(finetune=True)

    # Loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    optimizerFineTune = optim.SGD(modelFineTune.parameters(), lr=0.001, momentum=0.9)

    # Scheduler for learning rate decay
    expLrScheduler = optim.lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)


    modelFineTune = trainer.train(modelFineTune, criterion, optimizerFineTune, expLrScheduler, num_epochs=25)

    # Save model
    modelPath = dirs.saved_models + "test_mnist_resnet18.py"
    torch.save(modelFineTune.state_dict(), modelPath)

    print("Model after state dict:")
    for paramTensor in modelFineTune.state_dict():
        print(str(paramTensor).ljust(16), modelFineTune.state_dict()[paramTensor].size())
    
    print("Optimizer after state dict:")
    for varName in optimizerFineTune.state_dict():
        print(str(varName).ljust(16), modelFineTune.state_dict()[varName])
