import time
import numpy                as np
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from glob                   import glob

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import IndexManager
from libs.utils             import file_hash


indexPath  = Path(dirs.root) / "index" / "unlabeled_index_2019-8-16_11-37-59.csv"

indManager  = IndexManager(path=indexPath)

print("Index exists: ", indManager.indexExists)

elapsedTime = indManager.compute_frame_hashes()

print("Elapsed time: ", elapsedTime)

print(indManager.index.head())
print("Shape after")
print(indManager.index.shape)

print("Checking first hash:")
print(indManager.index.loc[0, "FrameHash"] == file_hash(indManager.index.loc[0, "FramePath"]))

newIndexPath = indexPath.with_name(indexPath.name+"_HASHES.csv")
indManager.write_index(dest_path=newIndexPath, prompt=False)