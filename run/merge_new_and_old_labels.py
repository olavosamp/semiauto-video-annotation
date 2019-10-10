import numpy                as np
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from copy                   import copy

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
import libs.commons         as commons

unlabeledPath = Path(dirs.iter_folder) / "full_dataset/iteration_2/unlabeled_images_iteration_2.csv"
oldPath       = Path(dirs.iter_folder) / "full_dataset/iteration_1/final_annotated_images_iteration_1.csv"
newPath       = Path(dirs.iter_folder) / "full_dataset/iteration_2/manual_labeled_index_raw_iteration_2.csv"

oldLabels = pd.read_csv(oldPath)
newLabels = pd.read_csv(newPath)

# Preprocessing: add FrameHash column
# Add folder path
datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_2/sampled_images/"
def _add_folder_path(path):
    path = datasetPath / Path(path)
    return str(path)

newLabels["FramePath"] = newLabels["imagem"].map(_add_folder_path)
newLabels["FrameHash"] = newLabels["FramePath"].map(utils.file_hash)

# Add annotation type column to manual_labeled_raw index
newLabels["Annotation"] = [commons.manual_annotation]*len(newLabels)

# Remove duplicates
oldLabels = dutils.remove_duplicates(oldLabels, "FrameHash")
newLabels = dutils.remove_duplicates(newLabels, "FrameHash")

# Get additional information for newLabels from main unlabeled index
# TODO: Do this as the second iteration step
unlabeledIndex = pd.read_csv(unlabeledPath)
unlabeledIndex = dutils.remove_duplicates(unlabeledIndex, "FrameHash")

newLabels = dutils.fill_index_information(unlabeledIndex, newLabels, "FrameHash")
print(newLabels.head())
print(newLabels.shape)

mergedIndex = pd.concat([newLabels, oldLabels], axis=0, sort=False)
print(mergedIndex.head())
print(mergedIndex.shape)

mergedIndex.to_csv(newPath.with_name("annotated_images_iteration_2.csv"), index=False)

