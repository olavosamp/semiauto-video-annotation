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
    # Dataset root folder
    # datasetPath = Path(dirs.iter_folder) / "test_loop/iteration_0/sample_images_sorted/"
    datasetPath = Path(dirs.iter_folder) / "test_loop/iteration_0/sampled_images_sorted/"
    numImgBatch = 2

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
                    transforms.Resize(256), # Pq 256?
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
        ])
    }
    # Split datasets in train and validation sets
    trainPercentage = 0.8
    valPercentage   = 0.2

    # # Should be run only once to split images in train and val folders
    # data_folder_split(datasetPath, [trainPercentage, valPercentage])
    # exit()

    # Load Dataset objects for train and val sets from folder
    sets = ['train', 'val']
    imageDataset = {}
    for phase in sets:
        f = datasetPath / phase
        imageDataset[phase] = datasets.ImageFolder(str(f), transform=dataTransforms[phase])

    datasetLen = len(imageDataset['train']) + len(imageDataset['val'])

    print(imageDataset['train'])
    exit()


    # trainSize = int(datasetLen*trainPercentage)
    valSize   = int(datasetLen*valPercentage)
    trainSize = datasetLen - valSize # This avoids dataset length mismatch due to rounding set sizes

    # Instantiate trainer object
    trainer = TrainModel()

    # Perform training
    trainer.load_data(imageDataset, num_examples_per_batch=numImgBatch)

    modelFineTune = trainer.define_model_resnet18(finetune=True)

    # Loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    optimizerFineTune = optim.SGD(modelFineTune.parameters(), lr=0.001, momentum=0.9)

    # Scheduler for learning rate decay
    expLrScheduler = optim.lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)

    modelFineTune = trainer.train(modelFineTune, criterion, optimizerFineTune, expLrScheduler, num_epochs=25)

    # Save model
    modelPath = dirs.saved_models + "test_iter_resnet18.pt"
    torch.save(modelFineTune.state_dict(), modelPath)

    # print("Model state dict after:")
    # for paramTensor in modelFineTune.state_dict():
    #     print(str(paramTensor).ljust(16), modelFineTune.state_dict()[paramTensor].size())
    
    # print("Optimizer state dict after:")
    # for varName in optimizerFineTune.state_dict():
    #     print(str(varName).ljust(16), modelFineTune.state_dict()[varName])
