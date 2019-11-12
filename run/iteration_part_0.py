import numpy                as np
import pandas               as pd
from pathlib                import Path
from glob                   import glob
from tqdm                   import tqdm

import libs.dirs            as dirs
import libs.commons         as commons
import libs.dataset_utils   as dutils

# Get inputs
rede        = input("\nEnter net number.\n")
targetClass = commons.rede1_positive

datasetName = "full_dataset_rede_{}".format(rede)
loopFolder        = Path(dirs.iter_folder) / datasetName
prevAnnotatedPath = loopFolder / "iteration_0/final_annotated_images_rede_{}.csv".format(rede-1)

# Create new labeled and unlabeled datasets from previous level annotated images
dutils.start_loop(prevAnnotatedPath, rede, targetClass, commons.net_target_column[rede-1])
