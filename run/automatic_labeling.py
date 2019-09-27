import os
import torch
import math
import random
import numpy                as np
import pandas               as pd
import matplotlib.pyplot       as plt
import torchvision.datasets as datasets
from PIL                    import Image
from pathlib                import Path
from torchvision            import transforms
from torch.utils.data       import DataLoader

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from models.trainer_class   import TrainModel
from libs.index             import IndexManager

outputPath      = Path(dirs.saved_models)/ "results_full_dataset_iteration_0_1000_epochs.pickle"
indexPath       = Path(dirs.iter_folder) / "full_dataset/iteration_0/unlabeled_images_iteration_1.csv"
indexDf         = pd.read_csv(indexPath)
indexDf.set_index("FrameHash", drop=False)

pickleData = utils.load_pickle(outputPath)

# Select only unlabeled data
# TODO: Change inference script to only perform inference in the unlabeled set
pickleData.set_index("ImgHashes", drop=False)
pickleData = pickleData.loc[indexDf.index]

outputs    = np.stack(pickleData["Outputs"])[:, 0]
outputs    = utils.normalize_array(outputs)
datasetLen = len(outputs)

idealUpperThresh = 0.392 # Ratio 95%
idealLowerThresh = 0.224 # Ratio 1%

print("\nAutomatic labeling with upper positive ratio 95%:")
_, _ = dutils.automatic_labeling(outputs, idealUpperThresh, idealLowerThresh)

idealUpperThresh = 0.416 # Ratio 99%
print("\nAutomatic labeling with upper positive ratio 99%:")
upperClassified, lowerClassified = dutils.automatic_labeling(outputs, idealUpperThresh, idealLowerThresh)

newLabels = np.concatenate([upperClassified, lowerClassified])
newLabeledIndex = indexDf.loc[newLabels, :]
print(newLabeledIndex.shape)