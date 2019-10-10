import numpy                as np
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from copy                   import copy

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils

unlabeledPath  = Path(dirs.index)       / "unlabeled_index_2019-8-18_19-32-37_HASHES.csv"
manualPath     = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images_iteration_0.csv"
autoPath       = Path(dirs.iter_folder) / "full_dataset/iteration_0/automatic_labeled_images_iteration_1.csv"

def fill_index_information(reference_index, to_fill_index):
    reference_index.set_index("FrameHash", drop=False, inplace=True)
    reference_index = reference_index[~reference_index.index.duplicated()]

    complete_index = reference_index.loc[to_fill_index.index, :]
    complete_index["rede1"] = to_fill_index["rede1"]
    complete_index["rede2"] = to_fill_index["rede2"]
    complete_index["rede3"] = to_fill_index["rede3"]
    complete_index["set"]   = to_fill_index["set"]
    return complete_index.copy()

manualIndex = pd.read_csv(manualPath)
autoIndex   = pd.read_csv(autoPath)

manualIndex.set_index("FrameHash", drop=False, inplace=True)
autoIndex.set_index("FrameHash", drop=False, inplace=True)

# Get additional information for manualIndex from main unlabeled index
unlabeledIndex = pd.read_csv(unlabeledPath)
manualIndex = fill_index_information(unlabeledIndex, manualIndex)


manualIndex = manualIndex[~manualIndex.index.duplicated()]
autoIndex   = autoIndex[~autoIndex.index.duplicated()]

manualLen   = len(manualIndex)
autoLen     = len(autoIndex)

manualIndex["Annotation"] = ["manual"]*manualLen
autoIndex["Annotation"]   = ["auto"]*autoLen


print("manual: ", manualIndex.shape)
print("auto:   ", autoIndex.shape)
print(manualIndex.head())
print(autoIndex.head())
# input()
mergedIndex = pd.concat([manualIndex, autoIndex], axis=0, sort=False)

print(mergedIndex.head())
print(mergedIndex.shape)
mergedIndex.to_csv(manualPath.with_name("final_annotated_images_iteration_1.csv"))