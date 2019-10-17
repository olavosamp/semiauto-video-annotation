import numpy                as np
import pandas               as pd
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from libs.index             import IndexManager

iteration   = 2
epochs      = 100
rede        = 1

savePath = Path(dirs.saved_models) / \
                    "results_full_dataset_iteration_{}_{}_epochs.pickle".format(iteration, epochs)
indexPath = Path(dirs.iter_folder) / \
                    "full_dataset/iteration_{}/unlabeled_images_iteration_{}.csv".format(iteration, iteration)
sampledImagesPath = Path(dirs.images) / "full_dataset_results_samples"

# Load model outputs and unlabeled images index
index = IndexManager(indexPath)
data  = utils.load_pickle(savePath)
datasetLen = len(data)

print(data.keys())
print(np.shape(data))

outputs = np.stack(data['Outputs'])[:, 0]
outputs = utils.normalize_array(outputs)

# Thresholds determined by find_threshold functions
idealUpperThresh = 0.392
idealLowerThresh = 0.224

indexes = np.arange(datasetLen, dtype=int)
upperClassIndex = indexes[np.greater(outputs, idealUpperThresh)]
lowerClassIndex = indexes[np.less(outputs, idealLowerThresh)]
totalClassified = len(upperClassIndex) + len(lowerClassIndex)

print("upperClassIndex: ", len(upperClassIndex))
print("lowerClassIndex: ", len(lowerClassIndex))
print("\nImages automatically labeled: {}/{} = {:.2f} %".format(totalClassified, datasetLen,
                                                            (totalClassified)/datasetLen*100))
# Randomly sample images
samplePercent = 0.1
# upperNum = round(len(upperClassIndex)*samplePercent)
# lowerNum = round(len(lowerClassIndex)*samplePercent)
upperNum = 7866
lowerNum = 9187

print("\nSampling {} images.".format(upperNum+lowerNum))
np.random.shuffle(upperClassIndex) # Is shuffling indexes equivalent to uniform sampling?
np.random.shuffle(lowerClassIndex)

upperClassIndexSampled = upperClassIndex[:upperNum]
lowerClassIndexSampled = lowerClassIndex[:lowerNum]

# Add a missing part of image paths
def add_unlabeled(filepath):
    s = Path(filepath)
    return str(s.parent / "unlabeled" / s.name)
index.index["FramePath"] = index.index["FramePath"].map(add_unlabeled)

# Select index entries by ImgHash
hashUpper = data["ImgHashes"][upperClassIndexSampled]
index.index.set_index('FrameHash', drop=False, inplace=True)
index.index = index.index.loc[hashUpper, :]
index.index.reset_index(drop=True, inplace=True)

index.copy_files(sampledImagesPath / "Evento")

index = IndexManager(indexPath)
hashLower = data["ImgHashes"][lowerClassIndexSampled]
index.index.set_index('FrameHash', drop=False, inplace=True)
index.index = index.index.loc[hashLower, :]
index.index.reset_index(drop=True, inplace=True)

index.copy_files(sampledImagesPath / "NaoEvento")
