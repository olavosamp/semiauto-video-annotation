import pandas               as pd
from pathlib                import Path

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import IndexManager
from libs.dataset_utils     import *


indexPath  = Path(dirs.iter_folder) / "test_loop/iteration_0/sampled_images.csv"

ind  = IndexManager(path=indexPath)
# print(ind.index.head())
# exit()
newLabelsPath = Path(dirs.iter_folder) / "test_loop/iteration_0/sampled_images_labels.csv"
labelsDf = add_frame_hash_to_labels_file(newLabelsPath, framePathColumn='imagem')

# Merge operation
ind.merge_annotations(newLabelsPath)

print(labelsDf.head())