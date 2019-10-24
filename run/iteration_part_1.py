import torch
import pandas                   as pd
import numpy                    as np
import torch.nn                 as nn
import torchvision.datasets     as datasets
from pathlib                    import Path

import libs.dirs                as dirs
import libs.commons             as commons
import libs.utils               as utils
import libs.dataset_utils       as dutils
import models.utils             as mutils
from libs.index                 import IndexManager
from libs.iteration_manager     import SampleImages


if __name__ == "__main__":
    seed           = 42
    iteration      = 2
    rede           = 1
    epochs         = 1000
    trainBatchSize = 256
    inferBatchSize = 64

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "full_dataset_softmax/iteration_{}/".format(iteration)

    previousIterFolder   = get_iter_folder(iteration-1)
    iterFolder           = get_iter_folder(iteration)
    nextIterFolder       = get_iter_folder(iteration+1)
    unlabeledIndexPath   = previousIterFolder / "unlabeled_images_iteration_{}.csv".format(iteration-1)

    ### Next Iteration
    ## Sample images for manual annotation
    sampler = SampleImages(unlabeledIndexPath, iterFolder, seed=seed)
    sampler.sample(percentage=0.01)
    # print(sampler.imageSourcePaths)
    print(sampler.imageSourcePaths.shape)
    # Sampled images index will be created during the manual annotation