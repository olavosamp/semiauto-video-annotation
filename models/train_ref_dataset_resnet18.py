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


if __name__ == "__main__":
    seed = None
    rede = int(input("\nEnter net number.\n"))
    numImgBatch = 256
    numEpochs   = 25

    modelPath = dirs.saved_models + \
            "reference_rede_{}_dataset_{}_epochs.pt".format(rede, numEpochs)
    historyPath = dirs.saved_models + \
            "history_reference_rede_{}_dataset_{}_epochs.pickle".format(rede, numEpochs)

    # Dataset root folder
    datasetPath = Path(dirs.dataset) / "reference_dataset_rede_{}".format(rede)

    # ImageNet statistics
    dataTransforms = mutils.resnet_transforms(commons.IMAGENET_MEAN, commons.IMAGENET_STD)

    # Load Dataset objects for train and val sets from folder
    sets = ['train', 'val']
    imageDataset = {}
    for phase in sets:
        f = datasetPath / phase
        imageDataset[phase] = datasets.ImageFolder(str(f),
                                                   transform=dataTransforms[phase],
                                                   is_valid_file=utils.check_empty_file)

    datasetLen = len(imageDataset['train']) + len(imageDataset['val'])

    history, modelFineTune = mutils.train_network(datasetPath, dataTransforms, epochs=numEpochs,
                                        batch_size=numImgBatch,
                                        model_path=modelPath,
                                        history_path=historyPath,
                                        seed=seed,
                                        weighted_loss=True,
                                        device_id=1)