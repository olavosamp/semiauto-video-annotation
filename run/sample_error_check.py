import numpy                as np
import pandas               as pd
from pathlib                import Path
from glob                   import glob
from tqdm                   import tqdm

import libs.dirs            as dirs
import libs.utils           as utils
# import libs.dataset_utils   as dutils
# from libs.index             import IndexManager

datasetName = "full_dataset_softmax"

sampledImagesPath = Path(dirs.images) / "{}_results_samples".format(datasetName)
loopFolder        = Path(dirs.iter_folder) / datasetName

# Get list of all annotated files
folderList = utils.make_path( glob(str(loopFolder)+"/iteration*"))

# Drop first and last folders, as they don't have automatic annotations
# Iterate in reverse order to leave largest files for last
autoFolderList = folderList[1:-1][::-1]
autoIndexList = []
for folder in tqdm(autoFolderList):
    iteration = str(folder)[-1]
    autoIndexList.append(pd.read_csv(folder/ "automatic_labeled_images_iteration_{}.csv".format(iteration)))
autoIndexFull = pd.concat(autoIndexList, axis=0)
print(autoIndexFull.shape)
