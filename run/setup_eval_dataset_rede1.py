import os
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

'''
/home/olavosamp/projetos/projeto_final/semiauto-video-annotation/run/
/home/common/flexiveis/datasets/handpicked/
'''
'''
    Script to set up rede1 dataset for training and evaluation.
'''
rede = 1
netName = 'rede'+str(rede)

referenceIndexPath  = Path(dirs.index) / "main_index_2019-7-1_18-22-41.csv"
remoteDatasetPath   = Path("/home/common/flexiveis/datasets/handpicked/")
semiautoDatasetPath = Path(dirs.dataset) / "semiauto_dataset_v1_rede_{}".format(rede)
semiautoIndexPath   = Path(dirs.dataset) / "semiauto_dataset_v1_rede_{}.csv".format(rede)
refDatasetPath      = Path(dirs.dataset) / "reference_dataset_rede_{}".format(rede)
datasetIndexPath    = Path(dirs.iter_folder) / "dataset_rede_{}_eval_setup.csv".format(rede)
trainTempPath       = Path(dirs.iter_folder) / "rede_{}_train_dataset.csv".format(rede)
valTempPath         = Path(dirs.iter_folder) / "rede_{}_val_dataset.csv".format(rede)

def _discard_middle_folders(path):
    path = Path(path).relative_to(remoteDatasetPath)
    image_set   = path.parts[0]
    image_class = path.parts[1]
    image_name  = path.name

    # Merge confusion to not_duct
    if image_class == "confusion":
        image_class = "not_duct"
    tailPath = [image_set, image_class, image_name]

    return refDatasetPath / "/".join(tailPath)

referenceIndex = pd.read_csv(referenceIndexPath, low_memory=False)

# Move images to new dataset location and discard middle folders
# dataset should look like this "...dataset/set/class/img.jpg"
if refDatasetPath.is_dir():
    input("\nDataset dest path already exists. Delete and overwrite?\n")
    sh.rmtree(refDatasetPath)
else:
    dirs.create_folder(refDatasetPath)

globString = str(remoteDatasetPath)+"/**/*jpg"
sourceList = glob(globString, recursive=True)
destList   = list(map(_discard_middle_folders, sourceList))

# Copy reference dataset and merge class confusion to not-duct
success = sum(list(map(utils.copy_files, sourceList, destList)))
print("\nMoved {}/{} files.\n".format(success, len(sourceList)))

# globStringVal   = str(remoteDatasetPath)+"/val/**/*jpg"
# globStringTrain = str(remoteDatasetPath)+"/train/**/*jpg"

# imageListVal   = glob(globStringVal, recursive=True)
# imageListTrain = glob(globStringTrain, recursive=True)
# print("\nTrain set: {} images.".format(len(imageListTrain)))
# print("Val set:   {} images.".format(len(imageListVal)))

# Get reference dataset validation video list
# videoList = commons.val_videos_reference_dataset_rede_1
# hashList = []
# for videoTuple in videoList:
#     part1 = str(videoTuple[0])
#     part3 = str(videoTuple[2])
    
#     if videoTuple[1] is not None:
#         part2     = "DVD-"+str(videoTuple[1])
#         videoPath = "/".join([part1, part2, part3])
#     else:
#         videoPath = "/".join([part1, part3])
    
#     videoHash = utils.compute_file_hash_list(videoPath, folder=dirs.base_videos)
#     print("\n", videoPath)
#     print(videoHash)
#     hashList.extend(videoHash)

# print("")
# for videoHash in hashList:
#     print(videoHash)
hashList = commons.val_videos_reference_dataset_rede_1_hashes

# Split dataset in val and train following reference dataset
trainIndex, valIndex = dutils.split_validation_set_from_video_list(datasetIndexPath,
                                                                   hashList, key_column="HashMD5")

# Count errors
trainErrors = 0
for i in range(len(trainIndex)):
    video = trainIndex.loc[i, "HashMD5"]
    if video in hashList:
        print(i, ": ", video)
        trainErrors += 1

valErrors = 0
for i in range(len(valIndex)):
    video = valIndex.loc[i, "HashMD5"]
    if video not in hashList:
        print(i, ": ", video)
        valErrors += 1

print("\nSemiauto dataset split:")
print("train set: {} images.".format(len(trainIndex)))
print("val set: {} images.".format(len(valIndex)))
print("\nErrors:")
print("Train: {}\nVal:\t{}".format(trainErrors, valErrors))
print("\nNaNs:")
print("Train: {}\nVal:\t{}".format(np.sum(trainIndex[netName].isna()), np.sum(valIndex[netName].isna())))

trainIndex.dropna(axis=0, subset=["HashMD5"], inplace=True)
valIndex.dropna(axis=0, subset=["HashMD5"], inplace=True)

print("\nNaNs:")
print("Train: {}\nVal:\t{}".format(np.sum(trainIndex[netName].isna()), np.sum(valIndex[netName].isna())))


dutils.df_to_csv(trainIndex, trainTempPath)
dutils.df_to_csv(valIndex, valTempPath)

input("\nMoving datasets to train folder.\nPress enter to continue.\n")
# Move dataset to training folder, split in train/val folders
dutils.copy_dataset_to_folder(trainTempPath, semiautoDatasetPath / "train", path_column="FramePath")
dutils.move_to_class_folders(trainTempPath, semiautoDatasetPath / "train", target_net=netName,
                                    target_class=None, move=True)

dutils.copy_dataset_to_folder(valTempPath, semiautoDatasetPath / "val", path_column="FramePath")
dutils.move_to_class_folders(valTempPath, semiautoDatasetPath / "val", target_net=netName,
                                    target_class=None, move=True)

# Save complete dataset index and delete temporary train and val indexes
trainIndex['set'] = ['train']*len(trainIndex)
valIndex['set']   = ['val']*len(valIndex)

evalDatasetIndex = pd.concat([trainIndex, valIndex], axis=0)

# Delete temporary files
os.remove(trainTempPath)
os.remove(valTempPath)

# Write dataset index to file
dutils.df_to_csv(evalDatasetIndex, semiautoIndexPath)