import os
import time
import copy
import torch
import matplotlib
import torchvision
import torch.nn             as nn
import numpy                as np
import torch.optim          as optim
import matplotlib.pyplot    as plt
from pathlib                import Path
from torch.optim            import lr_scheduler
from torchvision            import datasets, models, transforms

import libs.dirs            as dirs
from libs.utils             import *
from models.trainer_class   import TrainModel


def torch_imshow(gridInput, mean, std, title=None):
    gridInput = gridInput.numpy().transpose((1,2,0))

    gridInput = std*gridInput + mean
    gridInput = np.clip(gridInput, 0, 1)

    ax = plt.imshow(gridInput)
    plt.title(title)
    # plt.pause(0.01)
    # plt.imsave("../images/testgrid.png", gridInput)

if __name__ == "__main__":
    datasetPath = Path(dirs.dataset) / "torch/hymenoptera_data"

    mean    = [0.485, 0.456, 0.406]
    std     = [0.229, 0.224, 0.225]

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
    imageDatasets = {
        x: datasets.ImageFolder(str(datasetPath / x), dataTransforms[x]) for x in ['train', 'val']
    }


    # Get an image batch of the training set
    # inputs, classes = next(iter(dataloaders['train']))

    # Make a grid and display it
    # imgGrid = torchvision.utils.make_grid(inputs)
    # torch_imshow(imgGrid, mean, std, title=[classNames[x] for x in classes])
    # plt.show()

    # Instantiate trainer object
    trainer = TrainModel()
    
    # device = torch.device('cuda:0')
    modelFineTune = trainer.define_model_resnet18(finetune=True)

    criterion = nn.CrossEntropyLoss()

    # Set optimizer
    optimizerFineTune = optim.SGD(modelFineTune.parameters(), lr=0.001, momentum=0.9)

    # Scheduler for learning rate decay
    expLrScheduler = lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)

    # Perform training
    trainer.load_data(imageDatasets, num_examples_per_batch=4)
    modelFineTune = trainer.train(modelFineTune, criterion, optimizerFineTune, expLrScheduler, num_epochs=25)
