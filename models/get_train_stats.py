import numpy                as np
import torchvision.datasets as datasets
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import models.utils         as mutils
import libs.commons         as commons
from libs.vis_functions     import plot_confusion_matrix


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

    history, _ = mutils.train_network(dataset_path, dataTransforms, epochs=epochs,
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


if __name__ == "__main__":
    rede = int(input("\nEnter net number.\n"))
    numEvals    = 5
    numEpochs   = 25

    # Dataset root folder
#     datasetPath = Path(dirs.dataset) / "reference_dataset_rede_{}".format(rede)
    datasetPath = Path(dirs.dataset) / "semiauto_dataset_v1_rede_{}".format(rede)

    modelFolder = Path(dirs.saved_models) / \
            "{}_{}_epochs".format(datasetPath.stem, numEpochs)
    historyFolder = Path(dirs.saved_models) / \
            "history_{}_{}_epochs".format(datasetPath.stem, numEpochs)
    filePath = Path(dirs.results) / \
            "log_evaluation_{}_{}_epochs.txt".format(datasetPath.stem, numEpochs)
    confMatPath = Path(dirs.results) / \
            "confusion_matrix_{}.pdf".format(datasetPath.stem)

    valLoss = []
    valAcc  = []
    # Run function many times and save best results
    for i in range(numEvals):
        print("\nStarting run number {}/{}.\n".format(i+1, numEvals))
        modelPath = modelFolder / "model_run_{}.pt".format(i)
        historyPath = historyFolder / "history_run_{}.pickle".format(i)

        roundValLoss, roundValAcc, confMat = wrapper_train(numEpochs, modelPath, historyPath, datasetPath)

        valLoss.append(roundValLoss)

        classAcc = mutils.compute_class_acc(confMat)
        avgAcc = np.mean(classAcc)
        valAcc.append(roundValAcc)
        print("Debug\nAvg acc: {:.3f}".format(avgAcc))
        print("other acc: {:.3f}\n".format(roundValAcc))

        # Save best confusion matrix
        if np.argmin(valLoss) == i:
            bestConfMat = confMat

    printString = ""
    printString += "\nFinished training {} evaluation runs for dataset\n{}\n".format(numEvals, datasetPath)
    printString += "\nResulting statistics:\n\
    Val Loss:\n\
        Mean: {:.3f}\n\
        Std : {:.3f}\n\
    Val Avg Acc:\n\
        Mean: {:.5f}\n\
        Std   {:.5f}\n".format(np.mean(valLoss), np.std(valLoss),
                                np.mean(valAcc), np.std(valAcc))
    print(printString)
    with open(filePath, mode='w') as f:
        f.write(printString)

    plot_confusion_matrix(confMat, show=False, save_path=confMatPath)

    # print("Conf matrix:")
    # print(confMat)
