import os
import torch
import math
import numpy                as np
import torchvision.datasets as datasets
import torch.nn             as nn
import torch.optim          as optim
from pathlib                import Path
from torchvision            import models, transforms
from torch.utils.data       import DataLoader

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from models.trainer_class   import TrainModel


def check_empty_file(path):
    return os.path.getsize(path) > 0


if __name__ == "__main__":
    datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images/val/"
    savePath    = Path(dirs.saved_models)/ "results_full_dataset_iteration_0.pickle"
    
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

    # Get list of image paths from dataset folder
    dataset = datasets.ImageFolder(str(datasetPath), transform=dataTransforms, is_valid_file=check_empty_file)
    imageTupleList  = dataset.imgs
    labelList       = dataset.targets
    datasetLen      = len(imageTupleList)
    
    imagePathList  = np.array(dataset.imgs)[:, 0]
    
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                         batch_size=batchSize,
    #                                         shuffle=False, num_workers=4)

    imgLoader = dutils.IndexLoader(imagePathList, batch_size=batchSize, transform=dataTransforms, label_list=labelList)
    
    # for img, imgHash in imgLoader:
    #     # print(img)
    #     # print(np.shape(img))
    #     print(imgHash)
    #     # break

    # Instantiate trainer object
    trainer = TrainModel()
    trainer.numClasses = 2

    # Set model
    modelFineTune = trainer.define_model_resnet18(finetune=False)

    # Perform inference
    outputs, imgHashes, labels = trainer.model_inference(imgLoader)

    outputTuple = (outputs, imgHashes, labels)

    print()
    print(outputTuple[0][:20])
    # print(outputTuple[1][:20])
    print(np.shape(outputTuple))
    # input()

    # Save output to pickle file
    print("\nSaving outputs file to ", savePath)
    utils.save_pickle(outputTuple, savePath)
