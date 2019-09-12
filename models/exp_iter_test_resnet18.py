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
from libs.dataset_utils     import data_folder_split
from models.trainer_class   import TrainModel


if __name__ == "__main__":
    # Dataset root folder
    # datasetPath = Path(dirs.iter_folder) / "test_loop/iteration_0/sample_images_sorted/"
    datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images/"
    numImgBatch = 4
    numEpochs   = 15

    modelPath   = dirs.saved_models + "full_dataset_no_finetune.pt"
    historyPath = dirs.saved_models + "full_dataset_history_no_finetune.pickle"

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
    # # Split datasets in train and validation sets
    # trainPercentage = 0.8
    # valPercentage   = 0.2

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
