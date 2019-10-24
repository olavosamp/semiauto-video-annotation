import torch
import random
import numpy                as np
import pandas               as pd
import libs.dataset_utils   as dutils
import torch.nn             as nn
import torch.optim          as optim
import torchvision.datasets as datasets
from torchvision            import transforms

import libs.utils           as utils
from models.trainer_class   import TrainModel


## Pytorch utilities
def set_torch_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


## Model training and inference
def train_network(dataset_path, data_transforms, epochs=25, batch_size=64,
                  model_path="./model_weights.pt",
                  history_path="./train_history.pickle",
                  seed=None):
    if seed:
        set_torch_random_seeds(seed)

    # Load Dataset objects for train and val sets from folder
    sets = ['train', 'val']
    imageDataset = {}
    for phase in sets:
        f = dataset_path / phase
        imageDataset[phase] = datasets.ImageFolder(str(f),
                                                   transform=data_transforms[phase],
                                                   is_valid_file=utils.check_empty_file)

    # datasetLen = len(imageDataset['train']) + len(imageDataset['val'])

    # Instantiate trainer object
    trainer = TrainModel()

    # Perform training
    trainer.load_data(imageDataset, num_examples_per_batch=batch_size)

    modelFineTune = trainer.define_model_resnet18(finetune=False)

    # Set optimizer and Loss criterion
    modelFineTune = trainer.train(modelFineTune,
                                  nn.CrossEntropyLoss(),
                                  optim.Adam(modelFineTune.parameters()),
                                  scheduler=None, num_epochs=epochs)

    # Save train history and trained model weights
    if history_path:
        history = trainer.save_history(history_path)
    if model_path:
        torch.save(modelFineTune.state_dict(), model_path)
    
    return history, modelFineTune.state_dict()


def resnet_transforms(mean, std):
    '''
        Define default transforms for Resnet neural network.
    '''
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
        ])}
    return dataTransforms


def dataset_inference_val(dataset_path, data_transforms, model_path, save_path, batch_size=64, verbose=True):
    '''
        Perform inference on validation set and save outputs to file
    '''
    # Get list of image paths from dataset folder
    dataset = datasets.ImageFolder(str(dataset_path),
                                    transform=data_transforms,
                                    is_valid_file=utils.check_empty_file)
    imageTupleList  = dataset.imgs
    datasetLen      = len(imageTupleList)
    labelList       = dataset.targets
    
    imagePathList  = np.array(dataset.imgs)[:, 0]

    if verbose:
        print("\nDataset information: ")
        print("\t", datasetLen, "images.")
        print("\nClasses: ")
        for key in dataset.class_to_idx.keys():
            print("\t{}: {}".format(dataset.class_to_idx[key], key))

    imgLoader = dutils.IndexLoader(imagePathList, batch_size=batch_size,
                                   transform=data_transforms, label_list=labelList)

    # Instantiate trainer object
    trainer = TrainModel(model_path=model_path)
    trainer.numClasses = 2

    # Set model
    trainer.define_model_resnet18(finetune=False, print_summary=True)

    outputs, imgHashes, labels = trainer.model_inference(imgLoader)

    outputDf = pd.DataFrame({"Outputs":   outputs,
                             "ImgHashes": imgHashes,
                             "Labels":    labels})

    ## Save output to pickle file
    print("\nSaving outputs file to ", save_path)
    outputDf.to_pickle(save_path)
    return outputDf

