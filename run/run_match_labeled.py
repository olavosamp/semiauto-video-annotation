import numpy                as np
import pandas               as pd
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from libs.index             import IndexManager

unlabelIndexPath    = Path(dirs.index) / "unlabeled_index_2019-8-18_19-32-37_HASHES.csv"
sampledIndexPath    = Path(dirs.iter_folder)/ "full_dataset/iteration_0/sampled_images_iteration_1.csv"
newUnlabelIndexPath = Path(dirs.iter_folder)/ "full_dataset/iteration_0/unlabeled_images_iteration_1.csv"

# Load model outputs and unlabeled images index
indexUnlabel = IndexManager(unlabelIndexPath)
indexSampled = IndexManager(sampledIndexPath)

print(indexUnlabel.index.shape)

# Select index entries by ImgHash
indexUnlabel.index.set_index('FrameHash', drop=False, inplace=True)
indexUnlabel.index.drop(labels=indexSampled.index["FrameHash"], axis=0, inplace=True)

print(indexUnlabel.index.shape)

indexUnlabel.index.reset_index(drop=True, inplace=True)
indexUnlabel.write_index(newUnlabelIndexPath, prompt=False)

# indexSampled.index.set_index('FrameHash', drop=False, inplace=True)
# indexSampled.index.reset_index(drop=True, inplace=True)
