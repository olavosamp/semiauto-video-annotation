import numpy                as np
import pandas               as pd
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from libs.index             import IndexManager

savePath            = Path(dirs.saved_models)/ "results_full_dataset_iteration_0_1000_epochs.pickle"
indexPath           = Path(dirs.index) / "unlabeled_index_2019-8-18_19-32-37_HASHES_ATUAL.csv"
sampledImagesPath   = Path(dirs.images)/ "full_dataset_results_samples"

index = IndexManager(indexPath)
data  = utils.load_pickle(savePath)
datasetLen = len(data)

print(data.keys())
print(np.shape(data))

# print(index.index.head())
# exit()
outputs = np.stack(data['Outputs'])[:, 0]
outputs = utils.normalize_array(outputs)

idealUpperThresh = 0.392    # Determined by find_threshold functions
idealLowerThresh = 0.224

indexes = np.arange(datasetLen, dtype=int)
upperClassIndex = indexes[np.greater(outputs, idealUpperThresh)]
lowerClassIndex = indexes[np.less(outputs, idealLowerThresh)]
totalClassified = len(upperClassIndex) + len(lowerClassIndex)

print("upperClassIndex: ", len(upperClassIndex))
print("lowerClassIndex: ", len(lowerClassIndex))
print("\nImages automatically labeled: {}/{} = {:.2f} %".format(totalClassified, datasetLen,
                                                            (totalClassified)/datasetLen*100))

samplePercent = 0.1

upperNum = round(len(upperClassIndex)*samplePercent)
lowerNum = round(len(lowerClassIndex)*samplePercent)

np.random.shuffle(upperClassIndex)
np.random.shuffle(lowerClassIndex)

upperClassIndexSampled = upperClassIndex[:upperNum]
lowerClassIndexSampled = lowerClassIndex[:lowerNum]

# Adds a missing part of image paths
def add_unlabeled(filepath):
    s = Path(filepath)
    return str(s.parent / "unlabeled" / s.name)
index.index["FramePath"] = index.index["FramePath"].map(add_unlabeled)

hashUpper = data["ImgHashes"][upperClassIndexSampled]
index.index.set_index('FrameHash', drop=False, inplace=True)
index.index = index.index.loc[hashUpper, :]
index.index.reset_index(drop=True, inplace=True)

index.move_files(sampledImagesPath / "Evento")

index = IndexManager(indexPath)
hashLower = data["ImgHashes"][lowerClassIndexSampled]
index.index.set_index('FrameHash', drop=False, inplace=True)
index.index = index.index.loc[hashLower, :]
index.index.reset_index(drop=True, inplace=True)

index.move_files(sampledImagesPath / "NaoEvento")
