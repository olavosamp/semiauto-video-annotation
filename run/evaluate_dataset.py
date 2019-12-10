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

remoteDatasetPath = Path("/home/common/flexiveis/datasets/events_191016/val/")
datasetIndexPath  = Path(dirs.iter_folder) / "dataset_rede_3_positives_binary.csv"


def get_ref_dataset_val_video_list(folder_path, verbose=False):
    globString = str(folder_path)+"/**"
    folderList = glob(globString, recursive=True)
    videoList = []
    for pathEntry in folderList:
        relString = Path(pathEntry).relative_to(folder_path)
        if len(relString.parts) == 2:
            videoHash = relString.parts[-1]
            videoList.append(videoHash)
    videoList = list(set(videoList))

    return videoList

videoList = get_ref_dataset_val_video_list(remoteDatasetPath)
print(videoList)
print(len(videoList))

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

trainIndex.to_csv(Path(dirs.iter_folder) / "train_dataset.csv")
valIndex.to_csv(Path(dirs.iter_folder) / "val_dataset.csv")