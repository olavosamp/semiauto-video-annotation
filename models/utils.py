import os
import torch
import random
import numpy                as np
import pandas               as pd
import torch.nn             as nn
import torch.optim          as optim
import torchvision.datasets as datasets
from pathlib                import Path
from torchvision            import transforms

import libs.utils           as utils
import libs.dataset_utils   as dutils
import libs.dirs            as dirs
from libs.index             import IndexManager
from models.trainer_class   import TrainModel


## Pytorch utilities
def set_torch_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


## Model training and inference
def get_loss_weights(dataset, loss_function):
    '''
        Compute weights for class size unbalance compensation.

        The function computes class_weights, a vector of size C, where C is the
        number of classes in the dataset.
         
        Each element represents the weight of a class given by
        datasets.class_to_idx and is calculated by
        .. math::
            w_i &= \frac{l_C}{l_{C_i}}, i \in [1, \..., C] \\,
            w_0 &= 1 - \sum{w_i}_{i=1}^C \\,

        Arguments:
        dataset: dict of torchvision.datasets.ImageFolder
            Dict containing keys 'val' and 'train', which values are torchvision
            ImageFolder objects.

        loss_function: class constructor
            The loss function/class to be instantiated with the class weights. The class constructor 
            must accept (and use) the 'weight' keyword argument.
        
        Returns:
            Returns the loss_function input instantiated with class_weights.
    '''
    class_weights = []
    # TODO: Check class weight formula
    imgListTrain = np.stack(dataset['train'].imgs, axis=1)
    imgListVal   = np.stack(dataset['val'].imgs, axis=1)
    imgList = np.concatenate([imgListTrain, imgListVal], axis=0)

    print(imgListTrain.shape)
    print(imgListVal.shape)
    print(imgList.shape)
    input()

    class_sizes = {}
    for class_index in dataset['train'].class_to_idx.values():
        class_sizes[class_index] = np.sum(np.equal(imgList[:, 1], class_index))

    print(class_sizes)
    input()
    return loss_function(weight=class_weights)

def train_network(dataset_path, data_transforms, epochs=25, batch_size=64,
                  model_path="./model_weights.pt",
                  history_path="./train_history.pickle",
                  weighted_loss=True,
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
    if weighted_loss:
        loss = nn.CrossEntropyLoss()
    else:
        loss = get_loss_weights(imageDataset, nn.CrossEntropyLoss)

    optimizer = optim.Adam(modelFineTune.parameters())
    modelFineTune = trainer.train(modelFineTune,
                                  loss,
                                  optimizer,
                                  scheduler=None, num_epochs=epochs)

    # Save train history and trained model weights
    if model_path:
        dirs.create_folder(Path(model_path).parent)
        torch.save(modelFineTune.state_dict(), model_path)
    if history_path:
        dirs.create_folder(Path(history_path).parent)
        history = trainer.save_history(history_path)
    
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


def _model_inference(image_path_list, data_transforms, label_list, model_path, batch_size, seed=None):
    imgLoader = dutils.IndexLoader(image_path_list, batch_size=batch_size,
                                transform=data_transforms, label_list=label_list)

    # Instantiate trainer object
    trainer = TrainModel(model_path=model_path, seed=seed)
    trainer.numClasses = 2

    # Set model
    trainer.define_model_resnet18(finetune=False, print_summary=True)

    outputs, imgHashes, labels = trainer.model_inference(imgLoader)

    outputDf = pd.DataFrame({"Outputs":   outputs,
                            "ImgHashes": imgHashes,
                            "Labels":    labels})
    return outputDf


def dataset_inference_val(dataset_path, data_transforms, model_path, save_path, batch_size=64, force=False,
                            seed=None, verbose=True):
    '''
        Perform inference on validation set and save outputs to file.

        force: Boolean
            If force is False, search for an existing output file and use it, if it exists. If force is
        True or output file doesn't exist, compute dataset output and save to file.
    '''
    if os.path.isfile(save_path) and not(force):
        outputDf = utils.load_pickle(save_path)
        if len(outputDf) > 0:
            return outputDf

    # Get list of image paths from dataset folder
    dataset = datasets.ImageFolder(str(dataset_path),
                                    transform=data_transforms,
                                    is_valid_file=utils.check_empty_file)
    imageTupleList  = dataset.imgs
    datasetLen      = len(imageTupleList)
    labelList       = dataset.targets
    
    imagePathList  = np.array(dataset.imgs)[:, 0]

    if verbose:
        print("Validation set inference.")
        print("\nDataset information: ")
        print("\t", datasetLen, "images.")
        print("\nClasses: ")
        for key in dataset.class_to_idx.keys():
            print("\t{}: {}".format(dataset.class_to_idx[key], key))
    
    outputDf = _model_inference(imagePathList, data_transforms, labelList, model_path, batch_size)
    
    ## Save output to pickle file
    if verbose:
        print("\nSaving outputs file to ", save_path)
    outputDf.to_pickle(save_path)
    return outputDf


def dataset_inference_unlabeled(dataset_path, data_transforms, model_path, save_path, batch_size=64,
                                force=False, seed=None, verbose=True):
    '''
        Perform inference on an unlabeled dataset, using a csv Index file as reference.
        
        force: Boolean
            If force is False, search for an existing output file and use it, if it exists. If force is
        True or output file doesn't exist, compute dataset output and save to file.
    '''
    if os.path.isfile(save_path) and not(force):
        outputDf = utils.load_pickle(save_path)
        if len(outputDf) > 0:
            return outputDf

    unlabelIndex = IndexManager(dataset_path)

    # Drop duplicated files
    unlabelIndex.index = dutils.remove_duplicates(unlabelIndex.index, "FrameHash")

    # Drop missing or corrupt images
    unlabelIndex.index = dutils.check_df_files(unlabelIndex.index, utils.check_empty_file, "FramePath")
    
    imagePathList = unlabelIndex.index["FramePath"].values
    datasetLen    = len(imagePathList)

    if verbose:
        print("\nUnlabeled set inference")
        print("\nDataset information: ")
        print("\t", datasetLen, "images.")

    # Label list for an unlabeled dataset (bit of a hack? is there a better way?)
    labelList = np.zeros(datasetLen)

    outputDf = _model_inference(imagePathList, data_transforms, labelList, model_path, batch_size)

    ## Save output to pickle file
    if verbose:
        print("\nSaving outputs file to ", save_path)
    outputDf.to_pickle(save_path)
