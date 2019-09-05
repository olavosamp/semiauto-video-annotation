import torch
import numpy                as np
import torchvision.datasets as datasets
import torch.nn             as nn
import torch.optim          as optim
from torchvision            import models, transforms
from torch.utils.data       import random_split, TensorDataset

import libs.dirs            as dirs
from libs.utils             import *
from libs.dataset_utils     import data_folder_split
from models.trainer_class   import TrainModel


if __name__ == "__main__":

    datasetPath = Path(dirs.iter_folder) / "test_loop/iteration_0/sample_images_sorted_test/"
    numImgBatch = 4

    # ImageNet statistics
    # No need to normalize pixel range from [0, 255] to [0, 1] because
    # ToTensor transform already does that
    mean    = [0.485, 0.456, 0.406]#/255
    std     = [0.229, 0.224, 0.225]#/255

    # Idea: compare with actual data statistics
    
    # Set transforms
    dataTransforms = {
        'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
        ])
    }


    # Dataset loaders for train and val sets
    imageDataset = datasets.ImageFolder(datasetPath)

    # # Load MNIST dataset
    # imageDataset = datasets.MNIST(datasetPath, train=True, transform=None, download=True)
    datasetLen = len(imageDataset)

    # Split datasets in train and validation sets
    trainPercentage = 0.8
    valPercentage   = 0.2

    data_folder_split(datasetPath, [trainPercentage, valPercentage])
    exit()

    # trainSize = int(datasetLen*trainPercentage)
    valSize   = int(datasetLen*valPercentage)
    trainSize = datasetLen - valSize # This avoids dataset length mismatch due to rounding set sizes

    trainSet, valSet = random_split(imageDataset, [trainSize, valSize])
    trainSet         = TensorDataset(trainSet)#, transforms=dataTransforms['train'])
    valSet           = TensorDataset(valSet)#,   transforms=dataTransforms['val'])

    # print(len(trainSet))
    # print(len(valSet))
    # print(trainSet.imgs)
    # print(trainSet.img)
    # print(trainSet.classes)
    # exit()

    # dataset['train']  = trainSet.dataset
    # dataset['val']    = valSet.dataset

    # # Indexes of each image set in the original dataset
    # trainIndexes      = trainSet.indices
    # valIndexes        = valSet.indices

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

    modelFineTune = trainer.define_model_resnet18(finetune=True)

    # Loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    optimizerFineTune = optim.SGD(modelFineTune.parameters(), lr=0.001, momentum=0.9)

    # Scheduler for learning rate decay
    expLrScheduler = optim.lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)


    modelFineTune = trainer.train(modelFineTune, criterion, optimizerFineTune, expLrScheduler, num_epochs=25)

    # Save model
    modelPath = dirs.saved_models + "test_mnist_resnet18.pt"
    torch.save(modelFineTune.state_dict(), modelPath)

    # print("Model state dict after:")
    # for paramTensor in modelFineTune.state_dict():
    #     print(str(paramTensor).ljust(16), modelFineTune.state_dict()[paramTensor].size())
    
    # print("Optimizer state dict after:")
    # for varName in optimizerFineTune.state_dict():
    #     print(str(varName).ljust(16), modelFineTune.state_dict()[varName])
