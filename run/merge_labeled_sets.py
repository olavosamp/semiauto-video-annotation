import numpy                as np
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from copy                   import copy

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils

# TODO: Use versions in dutils
def fill_index_information(reference_index, to_fill_index, index_column, columns_to_keep):
    reference_index.set_index(index_column, drop=False, inplace=True)
    to_fill_index.set_index(index_column, drop=False, inplace=True)

    complete_index = reference_index.loc[to_fill_index.index, :]
    for col in columns_to_keep:
        complete_index[col] = to_fill_index[col]

    complete_index.reset_index(drop=True, inplace=True)
    reference_index.reset_index(drop=True, inplace=True)
    to_fill_index.reset_index(drop=True, inplace=True)
    return complete_index.copy()

def merge_labeled_sets(auto_df, manual_df):
    manualIndex["Annotation"] = ["manual"]*len(manualIndex)
    autoIndex["Annotation"]   = ["auto"]*len(autoIndex)

    mergedIndex = pd.concat([manualIndex, autoIndex], axis=0, sort=False)
    return mergedIndex.copy()


unlabeledPath  = Path(dirs.index)       / "unlabeled_index_2019-8-18_19-32-37_HASHES.csv"
manualPath     = Path(dirs.iter_folder) / "full_dataset/iteration_1/sampled_images_iteration_1.csv"
autoPath       = Path(dirs.iter_folder) / "full_dataset/iteration_1/automatic_labeled_images_iteration_1.csv"

manualIndex = pd.read_csv(manualPath)
autoIndex   = pd.read_csv(autoPath)

manualIndex = dutils.remove_duplicates(manualIndex, "FrameHash")
autoIndex   = dutils.remove_duplicates(autoIndex, "FrameHash")

# Get additional information for manualIndex from main unlabeled index
# TODO: Do this as the second iteration step
unlabeledIndex = pd.read_csv(unlabeledPath)
unlabeledIndex = dutils.remove_duplicates(unlabeledIndex, "FrameHash")

manualIndex = fill_index_information(unlabeledIndex, manualIndex, "FrameHash", ["rede1", "rede2", "rede3", "set"])
print(manualIndex.head())
print(manualIndex.shape)

mergedIndex = merge_labeled_sets(manualIndex, autoIndex)
print(mergedIndex.head())
print(mergedIndex.shape)

mergedIndex.to_csv(manualPath.with_name("final_annotated_images_iteration_1.csv"), index=False)
