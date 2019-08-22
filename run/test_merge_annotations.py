import pandas               as pd
from pathlib                import Path

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import IndexManager


indexPath  = Path(dirs.iter_folder) / "test_loop/iteration_0/sampled_images.csv"

ind  = IndexManager(path=indexPath)
# print(ind.index.head())
# exit()
newLabelsPath = Path(dirs.iter_folder) / "test_loop/iteration_0/sampled_images_labels.csv"
ind.merge_annotations(newLabelsPath)
