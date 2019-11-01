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
    iteration = int(input("Enter iteration number."))
    seed           = 42
    # iteration      = 3
    rede           = 1

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "full_dataset_softmax/iteration_{}/".format(iteration)

    previousIterFolder   = get_iter_folder(iteration-1)
    iterFolder           = get_iter_folder(iteration)
    unlabeledIndexPath   = previousIterFolder / "unlabeled_images_iteration_{}.csv".format(iteration-1)
    sampledImageFolder   = iterFolder / "sampled_images"

    dirs.create_folder(iterFolder)
    dirs.create_folder(sampledImageFolder)

    ## Next Iteration
    print("\nSTEP: Sample images for manual annotation.")
    # Sample images for manual annotation
    sampler = SampleImages(unlabeledIndexPath, iterFolder, seed=seed)
    sampler.sample(percentage=0.01)
    print(sampler.imageSourcePaths.shape)
    
    # Sampled images index will be created during the manual annotation

    print("Image sampling finished.\nYou may now annotate sampled_images folder with the\
         labeling interface and run next step.")