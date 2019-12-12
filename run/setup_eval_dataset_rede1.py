import numpy                as np
import pandas               as pd
import shutil               as sh
# from tqdm                   import tqdm
from glob                   import glob
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.commons         as commons
import libs.dataset_utils   as dutils
# # from libs.index             import IndexManager
'''
/home/olavosamp/projetos/projeto_final/semiauto-video-annotation/run/
/home/common/flexiveis/datasets/handpicked/
'''
'''
    Script to set up rede1 dataset for training and evaluation.
'''
referenceIndexPath  = Path(dirs.index) / "main_index_2019-7-1_18-22-41.csv"
remoteDatasetPath   = Path("/home/common/flexiveis/datasets/handpicked/")
semiautoDatasetPath = Path(dirs.dataset) / "semiauto_dataset_v1_rede_1"
refDatasetPath      = Path(dirs.dataset) / "reference_dataset_rede_1"
datasetIndexPath    = Path(dirs.iter_folder) / "dataset_rede_3_positives_binary.csv"
trainPath           = Path(dirs.iter_folder) / "train_dataset.csv"
valPath             = Path(dirs.iter_folder) / "val_dataset.csv"

def _discard_middle_folders(path):
    path = Path(path).relative_to(remoteDatasetPath)
    tailPath = list(path.parts[:1])
    tailPath.append(path.name)
    return refDatasetPath / "/".join(tailPath)

referenceIndex = pd.read_csv(referenceIndexPath)

# Move images to new dataset location and discard middle folders
# dataset should look like this "...dataset/set/class/img.jpg"
if refDatasetPath.is_dir():
    input("\nDataset dest path already exists. Delete and overwrite?\n")
    sh.rmtree(refDatasetPath)

globString = str(remoteDatasetPath)+"/**/*jpg"
sourceList = glob(globString, recursive=True)
destList   = list(map(_discard_middle_folders, sourceList))

# Move images
map(utils.copy_files, sourceList, destList)
exit()
globStringVal   = str(remoteDatasetPath)+"/val/**/*jpg"
globStringTrain = str(remoteDatasetPath)+"/train/**/*jpg"

imageListVal   = glob(globStringVal, recursive=True)
imageListTrain = glob(globStringTrain, recursive=True)
print("Val images found: ", len(imageListVal))
print("Train images found: ", len(imageListTrain))

hashListVal     = utils.compute_file_hash_list(imageListVal)
hashListTrain   = utils.compute_file_hash_list(imageListTrain)

referenceIndex.set_index(commons.FRAME_HASH_COL_NAME, drop=False, inplace=True)
indexVal        = referenceIndex.reindex(labels=hashListVal, axis=0)
indexTrain      = referenceIndex.reindex(labels=hashListTrain, axis=0)

# # Copy reference dataset
# utils.copy_folder_tree(remoteDatasetPath, refDatasetPath)



# # Get reference dataset validation video list
# videoList = dutils.get_ref_dataset_val_video_list(refDatasetPath / "val")

# # Split dataset in val and train following reference dataset
# trainIndex, valIndex = dutils.split_validation_set_from_video_list(datasetIndexPath,
#                                                                    videoList, key_column="HashMD5")

# # Count errors
# trainErrors = 0
# for i in range(len(trainIndex)):
#     video = trainIndex.loc[i, "HashMD5"]
#     if video in videoList:
#         print(i, ": ", video)
#         trainErrors += 1

# valErrors = 0
# for i in range(len(valIndex)):
#     video = valIndex.loc[i, "HashMD5"]
#     if video not in videoList:
#         print(i, ": ", video)
#         valErrors += 1

# print("\nErrors:")
# print("Train: {}\nVal:\t{}".format(trainErrors, valErrors))
# print("\nNaNs:")
# print("Train: {}\nVal:\t{}".format(np.sum(trainIndex['rede3'].isna()), np.sum(valIndex['rede3'].isna())))

# trainIndex.dropna(axis=0, subset=["HashMD5"], inplace=True)
# valIndex.dropna(axis=0, subset=["HashMD5"], inplace=True)

# print("\nNaNs:")
# print("Train: {}\nVal:\t{}".format(np.sum(trainIndex['rede3'].isna()), np.sum(valIndex['rede3'].isna())))

# dutils.df_to_csv(trainIndex, trainPath)
# dutils.df_to_csv(valIndex, valPath)

# input("\nMoving datasets to train folder.\nPress enter to continue.\n")
# # Move dataset to training folder, split in train/val folders
# dutils.copy_dataset_to_folder(trainPath, semiautoDatasetPath / "train", path_column="FramePath")
# dutils.move_to_class_folders(trainPath, semiautoDatasetPath / "train", target_net="rede3",
#                                     target_class=None, move=True)

# dutils.copy_dataset_to_folder(valPath, semiautoDatasetPath / "val", path_column="FramePath")
# dutils.move_to_class_folders(valPath, semiautoDatasetPath / "val", target_net="rede3",
#                                     target_class=None, move=True)