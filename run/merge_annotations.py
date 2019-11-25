import numpy                as np
import pandas               as pd
from glob                   import glob
from copy                   import copy
from tqdm                   import tqdm
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
# from libs.index             import IndexManager

datasetName = "full_dataset_rede_2"

sampledImagesPath           = Path(dirs.images) / "{}_results_samples".format(datasetName)
loopFolder                  = Path(dirs.iter_folder) / datasetName
originalUnlabeledIndexPath  = loopFolder / "iteration_0/reference_images.csv"
compiledAutoIndexPath       = loopFolder / "final_automatic_images_{}.csv".format(datasetName)
compiledManualIndexPath     = loopFolder / "final_manual_images_{}.csv".format(datasetName)
annotatedIndexFullPath      = loopFolder / "final_annotated_images_{}.csv".format(datasetName)

originalUnlabeledIndex = pd.read_csv(originalUnlabeledIndexPath)

# Get list of all iteration folders
folderList = utils.make_path( glob(str(loopFolder)+"/iteration*"))
tempList = copy(folderList)
for path in tempList:
    if not(path.is_dir()):
        folderList.remove(path)

# Sort folder list by iteration
def _get_iter(path):
    return int(str(path).split("_")[-1])
folderList.sort(key=_get_iter)

# Group all automatic annotations
# Drop first and last iterations, as they don't have automatic annotations
iterList = list(range(len(folderList)))
autoIterList = iterList[1:-1]
# print(iterList)
# print(autoIterList)
# exit()
autoIndexList = []
for i in tqdm(autoIterList):
    folder = folderList[i]
    autoIndexList.append(pd.read_csv(folder/ "automatic_labeled_images_iteration_{}.csv".format(i)))

# tot = 0
# for i in range(len(autoIndexList)):
#     # print(i+1)
#     index = autoIndexList[i]
#     print("iteration_{}: {} images".format(i+1, index.shape[0]))
#     tot += index.shape[0]
# print("Total: ", tot)
# exit()

autoIndexFull = pd.concat(autoIndexList, axis=0, sort=False)

autoIndexFull = dutils.remove_duplicates(autoIndexFull, "FrameHash")
autoIndexFull.to_csv(compiledAutoIndexPath, index=False)
print(autoIndexFull.shape)

# Group all manual annotations
# Get cumulative manual index of second to last iteration (the last one with cumulative annotations)
cumManualIndex = pd.read_csv(folderList[-2] / \
                    "manual_annotated_images_iteration_{}_train_val_split.csv".format(iterList[-2]))

# Process sampled image csv
# Fill index information of sampled images of the final iteration
lastFolder = folderList[-1]
sampledLastIterIndex = pd.read_csv(lastFolder / "sampled_images_iteration_{}.csv".format(iterList[-1]))
sampledLastIterIndex["FrameHash"] = utils.compute_file_hash_list(sampledLastIterIndex["imagem"].values,
                                                        folder= lastFolder / "sampled_images")
manualLastIterIndex  = dutils.fill_index_information(originalUnlabeledIndex, sampledLastIterIndex,
                                        "FrameHash", [ 'rede1', 'rede2', 'rede3'])

manualIndexFull = pd.concat([cumManualIndex, manualLastIterIndex], axis=0, sort=False)
print(manualIndexFull.shape)
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