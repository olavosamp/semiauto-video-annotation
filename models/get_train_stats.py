import torch
import numpy                as np
import torch.nn             as nn
import torch.optim          as optim
import torchvision.datasets as datasets
from pathlib                import Path
from torchvision            import transforms

import libs.dirs            as dirs
import libs.utils           as utils
import models.utils         as mutils
import libs.commons         as commons
from models.trainer_class   import TrainModel


def wrapper_train(epochs, model_path, history_path, dataset_path):
    seed = None
    numImgBatch = 256
    use_weights = True

    # ImageNet statistics
    dataTransforms = mutils.resnet_transforms(commons.IMAGENET_MEAN, commons.IMAGENET_STD)

    # Load Dataset objects for train and val sets from folder
    sets = ['train', 'val']
    imageDataset = {}
    for phase in sets:
        f = dataset_path / phase
        imageDataset[phase] = datasets.ImageFolder(str(f),
                                                   transform=dataTransforms[phase],
                                                   is_valid_file=utils.check_empty_file)

    # datasetLen = len(imageDataset['train']) + len(imageDataset['val'])

    history, modelFineTune = mutils.train_network(dataset_path, dataTransforms, epochs=epochs,
                                        batch_size=numImgBatch,
                                        model_path=model_path,
                                        history_path=history_path,
                                        seed=seed,
                                        weighted_loss=use_weights,
                                        device_id=1)
    
    # Get best epoch results
    bestValIndex = np.argmin(history['loss-val'])
    bestValLoss  = history['loss-val'][bestValIndex]
    bestValAcc   = history['acc-val'][bestValIndex]
    return bestValLoss, bestValAcc


if __name__ == "__main__":
    rede = int(input("\nEnter net number.\n"))
    numEvals    = 10
    numEpochs   = 25

    modelFolder = Path(dirs.saved_models) / \
            "reference_rede_{}_dataset_{}_epochs".format(rede, numEpochs)
    historyFolder = Path(dirs.saved_models) / \
            "history_reference_rede_{}_dataset_{}_epochs".format(rede, numEpochs)

    # Dataset root folder
    datasetPath = Path(dirs.dataset) / "reference_dataset_rede_{}".format(rede)

    valLoss = []
    valAcc  = []
    # Run function many times and save best results
    for i in range(numEvals):
        print("\nStarting run number {}/{}.\n".format(i+1, numEvals))
        modelPath   = modelFolder / "model_run_{}.pt".format(i)
        historyPath = historyFolder / "history_run_{}.pickle".format(i)

        bestValLoss, bestValAcc = wrapper_train(numEpochs, modelPath, historyPath, datasetPath)
        
        valLoss.append(bestValLoss)
        valAcc.append(bestValAcc)


    # print(valLoss)
    # print(valAcc)
    print("\nFinished training {} evaluation runs for dataset\n{}.".format(numEvals, datasetPath))
    print("\nResulting statistics:\n\
    Val Loss:\n\
        Mean: {:.3f}\n\
        Std : {:.3f}\n\
    Val Acc:\n\
        Mean: {:.3f}\n\
        Std   {:.3f}\n".format(np.mean(valLoss), np.std(valLoss),
                                np.mean(valAcc), np.std(valAcc)))