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
    device_id = 0
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
                                        device_id=device_id)

    # Get best epoch results
    bestValIndex = np.argmin(history['loss-val'])
    bestValLoss  = history['loss-val'][bestValIndex]
    bestValAcc   = history['acc-val'][bestValIndex]
    confMat      = history['conf-val'][bestValIndex]
    return bestValLoss, bestValAcc, confMat

def compute_class_acc(confusion_matrix):
    '''
        confusion_matrix: array of floats
        Square array of floats of side n >= 2. Columns are assumed to contain true labels and rows,
        predicted labels, such as the sum of elements in of column i will be the number of true members
        of class i.
    '''
    assert np.shape(confusion_matrix)[0] == np.shape(confusion_matrix)[1] \
           and len(np.shape(confusion_matrix)) == 2, "Input must be a square matrix."
    
    numClasses = np.shape(confusion_matrix)[0]
    accList = []
    for i in range(numClasses):
        classAcc = confusion_matrix[i, i]/np.sum(confusion_matrix[i, :])
        accList.append(classAcc)
    
    return accList

if __name__ == "__main__":
    rede = int(input("\nEnter net number.\n"))
    numEvals    = 10
    numEpochs   = 25

    # Dataset root folder
    datasetPath = Path(dirs.dataset) / "reference_dataset_rede_{}".format(rede)
    # datasetPath = Path(dirs.dataset) / "semiauto_dataset_v1_rede_{}".format(rede)

    modelFolder = Path(dirs.saved_models) / \
            "{}_{}_epochs".format(datasetPath.stem, numEpochs)
    historyFolder = Path(dirs.saved_models) / \
            "history_{}_{}_epochs".format(datasetPath.stem, numEpochs)
    filePath = Path(dirs.results) / \
            "log_evaluation_{}_{}_epochs.txt".format(datasetPath.stem, numEpochs)

    valLoss = []
    valAcc  = []
    # Run function many times and save best results
    for i in range(numEvals):
        print("\nStarting run number {}/{}.\n".format(i+1, numEvals))
        modelPath = modelFolder / "model_run_{}.pt".format(i)
        historyPath = historyFolder / "history_run_{}.pickle".format(i)

        roundValLoss, roundValAcc, confMat = wrapper_train(numEpochs, modelPath, historyPath, datasetPath)

        valLoss.append(roundValLoss)

        if rede == 3:
            classAcc = compute_class_acc(confMat)
            avgAcc = np.mean(classAcc)
            valAcc.append(avgAcc)
            # print("Debug\nAvg acc: {:.3f}".format(avgAcc))
            # print("skm acc: {:.3f}\n".format(roundValAcc))
        else:
            valAcc.append(roundValAcc)

    printString = ""
    printString += "\nFinished training {} evaluation runs for dataset\n{}\n.".format(numEvals, datasetPath)
    printString += "\nResulting statistics:\n\
    Val Loss:\n\
        Mean: {:.3f}\n\
        Std : {:.3f}\n\
    Val Acc:\n\
        Mean: {:.3f}\n\
        Std   {:.3f}\n".format(np.mean(valLoss), np.std(valLoss),
                                np.mean(valAcc), np.std(valAcc))
    print(printString)
    with open(filePath, mode='w') as f:
        f.write(printString)

    # print("Conf matrix:")
    # print(confMat)
