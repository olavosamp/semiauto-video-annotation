import torch
import numpy                as np
import torchvision.datasets as datasets
import torch.nn             as nn
import torch.optim          as optim
from pathlib                import Path
from torchvision            import models, transforms
from torch.utils.data       import random_split, TensorDataset

import libs.dirs            as dirs
# from libs.utils             import *
from models.trainer_class   import TrainModel


if __name__ == "__main__":
    numImgBatch = 64
    numEpochs   = 25

    modelPath   = dirs.saved_models + "full_dataset_no_finetune_{}_epochs.pt".format(numEpochs)
    historyPath = dirs.saved_models + "history_full_dataset_no_finetune{}_epochs.pickle".format(numEpochs)

    # Dataset root folder
    # Images should have paths following
    #   root/classA/img.jpg
    #   root/classB/img.jpg
    #   ...
    datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images/"

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

    # Load Dataset objects for train and val sets from folder
    sets = ['train', 'val']
    imageDataset = {}
    for phase in sets:
        f = datasetPath / phase
        imageDataset[phase] = datasets.ImageFolder(str(f), transform=dataTransforms[phase])

    datasetLen = len(imageDataset['train']) + len(imageDataset['val'])

    # Instantiate trainer object
    trainer = TrainModel()

    # Perform training
    trainer.load_data(imageDataset, num_examples_per_batch=numImgBatch)

    modelFineTune = trainer.define_model_resnet18(finetune=False)

    # Loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    optimizerFineTune = optim.Adam(modelFineTune.parameters())

    # Scheduler for learning rate decay
    expLrScheduler = optim.lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)

    modelFineTune = trainer.train(modelFineTune, criterion, optimizerFineTune, scheduler=None, num_epochs=numEpochs)
    history = trainer.save_history(historyPath)

    # Save model
    torch.save(modelFineTune.state_dict(), modelPath)
