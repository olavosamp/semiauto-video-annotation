import numpy                as np
import pandas               as pd
# import cv2
# from tqdm                   import tqdm
from glob                   import glob
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.commons         as commons
import libs.dataset_utils   as dutils
# # from libs.index             import IndexManager
# /home/olavosamp/projetos/projeto_final/semiauto-video-annotation/run/
# /home/common/flexiveis/datasets/events_191016/val/**
# /home/common/flexiveis/datasets/events_191016/val/

'''
    Script to set up rede3 dataset for training and evaluation.
'''
remoteDatasetPath   = Path("/home/common/flexiveis/datasets/events_191016/val/")
datasetImagePath    = Path(dirs.dataset) / "new_dataset"
datasetIndexPath    = Path(dirs.iter_folder) / "dataset_rede_3_positives_binary.csv"
trainPath           = Path(dirs.iter_folder) / "train_dataset.csv"
valPath             = Path(dirs.iter_folder) / "val_dataset.csv"

videoList = dutils.get_ref_dataset_val_video_list(remoteDatasetPath)
# print(videoList)
# print(len(videoList))

trainIndex, valIndex = dutils.split_validation_set_from_video_list(datasetIndexPath,
                                                                   videoList, key_column="HashMD5")

trainErrors = 0
for i in range(len(trainIndex)):
    video = trainIndex.loc[i, "HashMD5"]
    if video in videoList:
        print(i, ": ", video)
        trainErrors += 1

valErrors = 0
for i in range(len(valIndex)):
    video = valIndex.loc[i, "HashMD5"]
    if video not in videoList:
        print(i, ": ", video)
        valErrors += 1

print("\nErrors:")
print("Train: {}\nVal:\t{}".format(trainErrors, valErrors))
print("\nNaNs:")
print("Train: {}\nVal:\t{}".format(np.sum(trainIndex['rede3'].isna()), np.sum(valIndex['rede3'].isna())))

trainIndex.dropna(axis=0, subset=["HashMD5"], inplace=True)
valIndex.dropna(axis=0, subset=["HashMD5"], inplace=True)

print("\nNaNs:")
print("Train: {}\nVal:\t{}".format(np.sum(trainIndex['rede3'].isna()), np.sum(valIndex['rede3'].isna())))

dutils.df_to_csv(trainIndex, trainPath)
dutils.df_to_csv(valIndex, valPath)

dutils.move_dataset_to_folder(trainPath, datasetImagePath / "train", path_column="FramePath")
dutils.move_to_class_folders(trainPath, datasetImagePath / "train", target_net="rede3",
                                    target_class=None, move=True)

dutils.move_dataset_to_folder(valPath, datasetImagePath / "val", path_column="FramePath")
dutils.move_to_class_folders(valPath, datasetImagePath / "val", target_net="rede3",
                                    target_class=None, move=True)