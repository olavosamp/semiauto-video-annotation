import numpy                as np
import pandas               as pd
from pathlib                import Path
from glob                   import glob
from tqdm                   import tqdm

import libs.dirs            as dirs
import libs.commons         as commons
import libs.dataset_utils   as dutils


# Get inputs
rede        = int(input("\nEnter net number.\n"))

if rede == 2:
    baseClass = commons.rede1_positive
    datasetName       = "full_dataset_rede_{}".format(rede)
elif rede == 3:
    baseClass = commons.rede2_positive
    event_class = dutils.get_input_target_class(commons.rede3_classes)
    datasetName       = "full_dataset_rede_{}_{}".format(rede, event_class.lower())
else:
    raise NotImplementedError("Only implemented for rede2 and rede3.")

loopFolder        = Path(dirs.iter_folder) / datasetName
prevAnnotatedPath = loopFolder / "iteration_0/final_annotated_images_rede_{}.csv".format(rede-1)

# Create new labeled and unlabeled datasets from previous level annotated images
dutils.start_loop(prevAnnotatedPath, baseClass, commons.net_target_column[rede-1])
