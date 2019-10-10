import numpy                as np
import pandas               as pd
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from libs.index             import IndexManager
''' Add FrameHash and FramePath to a interface-style csv annotations file.'''

folderPath          = Path()
datasetPath         = Path(dirs.iter_folder)/ "full_dataset/iteration_1/sampled_images"
sampledIndexPath    = Path(dirs.iter_folder)/ "full_dataset/iteration_1/olavo_uniformsampling_4676_corrections_train_val_split.csv"
newLabeledIndexPath = Path(dirs.iter_folder)/ "full_dataset/iteration_1/sampled_images_iteration_1.csv"

# Add folder path
def _add_folder_path(path):
    path = datasetPath / Path(path)
    return str(path)

# Load model outputs and unlabeled images index
indexSampled = IndexManager(sampledIndexPath)

indexSampled.index["FramePath"] = indexSampled.index["imagem"].map(_add_folder_path)

eTime = indexSampled.compute_frame_hashes(reference_column="FramePath")

print(eTime)
print(indexSampled.index.loc[:20, "FrameHash"])
print(indexSampled.index.shape)
print(indexSampled.index.head())

# indexSampled.index.set_index('FrameHash', drop=False, inplace=True)
# indexSampled.index.reset_index(drop=True, inplace=True)
indexSampled.write_index(dest_path=newLabeledIndexPath, make_backup=False, prompt=False)