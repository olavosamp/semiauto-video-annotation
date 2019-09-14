import os
import torch
import numpy                as np
import torchvision.datasets as datasets
import torch.nn             as nn
import torch.optim          as optim
from pathlib                import Path
from torchvision            import models, transforms
# from torch.utils.data       import random_split, TensorDataset

import libs.dirs            as dirs
from libs.dataset_utils     import data_folder_split
from models.trainer_class   import TrainModel


if __name__ == "__main__":
    batchSize = 32

    # ImageNet statistics
    # No need to normalize pixel range from [0, 255] to [0, 1] because
    # ToTensor transform already does that
    mean    = [0.485, 0.456, 0.406]#/255
    std     = [0.229, 0.224, 0.225]#/255
    
    dataTransforms = transforms.Compose([
                        transforms.Resize(256), # Pq 256?
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

    # datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images/"
    datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images_unsorted"

    def check_file_size(path):
        return os.path.getsize(path) > 0
    dataset = datasets.ImageFolder(datasetPath, transform=dataTransforms, is_valid_file=check_file_size)
    print(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batchSize,
                                            shuffle=False, num_workers=4)

    # Instantiate trainer object
    trainer = TrainModel()
    trainer.numClasses = 2

    # Perform training
    # inputDataset = datasets.ImageFolder(datasetPath, transform=dataTransforms['val'])

    modelFineTune = trainer.define_model_resnet18(finetune=False)
    trainer.model_inference(dataloader)
