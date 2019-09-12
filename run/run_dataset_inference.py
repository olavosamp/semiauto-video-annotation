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
    dataTransforms = {
        'val': transforms.Compose([
                    transforms.Resize(256), # Pq 256?
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
        ])
    }
    datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images/"
    
    # Instantiate trainer object
    trainer = TrainModel()

    # Perform training
    inputDataset = datasets.ImageFolder(datasetPath, transform=dataTransforms['val'])

    modelFineTune = trainer.define_model_resnet18(finetune=False)
