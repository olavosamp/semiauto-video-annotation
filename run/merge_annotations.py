import numpy                as np
import pandas               as pd
from pathlib                import Path
from glob                   import glob
from tqdm                   import tqdm

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
# from libs.index             import IndexManager

datasetName = "full_dataset_softmax"

sampledImagesPath = Path(dirs.images) / "{}_results_samples".format(datasetName)
loopFolder        = Path(dirs.iter_folder) / datasetName
unlabeledIndexPath      = loopFolder / "iteration_0/unlabeled_images_iteration_0_no_dups.csv"
compiledAutoIndexPath   = loopFolder / "final_automatic_images.csv"
compiledManualIndexPath = loopFolder / "final_manual_images.csv"
annotatedIndexFullPath  = loopFolder / "final_annotated_images.csv"

unlabeledIndex = pd.read_csv(unlabeledIndexPath)

# Get list of all iteration folders
folderList = utils.make_path( glob(str(loopFolder)+"/iteration*"))

# Group all automatic annotations
# Drop first and last folders, as they don't have automatic annotations
# Iterate in reverse order to leave largest files for last
autoFolderList = folderList[1:-1][::-1]
autoIndexList = []
for folder in tqdm(autoFolderList):
    iteration = str(folder)[-1]
    autoIndexList.append(pd.read_csv(folder/ "automatic_labeled_images_iteration_{}.csv".format(iteration)))
autoIndexFull = pd.concat(autoIndexList, axis=0, sort=False)

autoIndexFull = dutils.remove_duplicates(autoIndexFull, "FrameHash")
# autoIndexFull.to_csv(compiledAutoIndexPath, index=False)

# Group all manual annotations
cumManualIndex = pd.read_csv(folderList[-2] / "manual_annotated_images_iteration_8_train_val_split.csv")
# Process sampled image csv
sampledLastIterIndex = pd.read_csv(folderList[-1] / "sampled_images_iteration_9.csv")
sampledLastIterIndex["FrameHash"] = utils.compute_file_hash_list(sampledLastIterIndex["imagem"].values,
                                                        folder= folderList[-1] / "sampled_images")
manualLastIterIndex  = dutils.fill_index_information(unlabeledIndex, sampledLastIterIndex,
                                        "FrameHash", [ 'rede1', 'rede2', 'rede3'])


manualIndexFull = pd.concat([cumManualIndex, manualLastIterIndex], axis=0, sort=False)


manualIndexFull.to_csv(compiledManualIndexPath, index=False)

# Add Annotation column to indexes
autoIndexFull["Annotation"] = ['auto']*len(autoIndexFull)
manualIndexFull["Annotation"] = ['manual']*len(manualIndexFull)

print(autoIndexFull.head())
print(manualIndexFull.head())

annotatedIndexFull = pd.concat([manualIndexFull, autoIndexFull], axis=0, sort=False)
print(annotatedIndexFull.shape)
annotatedIndexFull = dutils.remove_duplicates(annotatedIndexFull, "FrameHash")
print(annotatedIndexFull.shape)
annotatedIndexFull.to_csv(annotatedIndexFullPath, index=False)