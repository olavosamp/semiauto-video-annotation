import re
import numpy        as np
import pandas       as pd
from pathlib        import Path
from glob           import glob

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager
from libs.utils     import string_list_complement, add_ok

# datasetPath = dirs.febe_base_videos
datasetPath = dirs.base_videos
datasetPath = "../20170724_FTP83G_Petrobras/"

def format_path(x): return Path(str(x).replace("\\", "/").replace(" ", "_"))

def format_relative_to(x):
    # If x starts with datasetPath, return relative path
    # TODO: Find a way to treat instances where datasetPath is not present without exceptions.
    try:
        return x.relative_to(dirs.base_videos)
    except ValueError:
        pass
    
    try:
        return x.relative_to(dirs.febe_base_videos)
    except ValueError:
        return x


def get_video_list_glob(datasetPath):
    videoList = []
    for format in commons.videoFormats:
        globString = datasetPath + "**" + "/*."+format
        globList = glob(globString, recursive=True)
        videoList.extend(globList)
    return videoList


def format_video_list(videoList):
    '''
       Find and format video paths in input dataset folder.
       Formatting includes converting all filepaths to Path objects;
        replacing backwards dashes with foward dashes and spaces with underscores;
        set filepaths relative to dataset path.
        Also deletes video paths VIDEO_TS that indicates DVD headers.
    '''
    # Format each path as a Path object
    videoList = list(map(format_path, videoList))
    # Format videos as relative to dataset path
    videoList = list(map(format_relative_to, videoList))

    # Delete DVD headers
    def matchVideoTS(x): return not(x.match("VIDEO_TS.VOB"))    # Map function
    mask      = list(map(matchVideoTS, videoList))              # Get index mask
    videoList = np.array(videoList)[mask]                       # Make path list into numpy array and
                                                                # apply index mask

    # Add _OK to videos without it
    videoList = add_ok(videoList)

    # Remove duplicated elements
    videoList = list(dict.fromkeys(videoList))
    
    # Format as Path again
    videoList = list(map(format_path, videoList))
    return videoList


mainIndexPath = Path(dirs.index)
newIndexPath  = Path(dirs.root) / "index" / "unlabeled_index.csv"

indLabeled  = IndexManager(path=mainIndexPath)            # Existing image index
# ind2  = IndexManager(path=newIndexPath)                   # New image index

# Get video paths in Main Index (labeled videos)
labeledVideos = list(indLabeled.index.VideoPath)
labeledVideos = format_video_list(labeledVideos)

for video in labeledVideos:
    print(video)
print("Labeled videos: ", len(labeledVideos))
print("\n")

## Get entire dataset video list
# # Get list from *Video Dataset Folder*
# allVideosList = get_video_list_glob(datasetPath)
# allVideos = format_video_list(allVideosList)

# # Get list from *Unlabeled Index*
allVideosPath = Path("index/unlabeled_index_2019-7-11_2-36-59.csv")
indAll = IndexManager(path=allVideosPath)

allVideosRaw = indAll.index.VideoPath
print("Processing file with {} entries.".format(np.shape(allVideosRaw)[0]))
allVideos    = format_video_list(allVideosRaw)

for video in allVideos:
    print(video)
print("Total videos: ", len(allVideos))
print("\n")

# Get video paths that are unlabeled (all - labeled)
unlabeledVideos = string_list_complement(allVideos, labeledVideos)
for video in unlabeledVideos:
    print(video)
print("Unlabeled videos: {}, should be {}.".format(len(unlabeledVideos), len(allVideos) - len(labeledVideos)))
print("\n")


## Miscellaneous string experiments
# print("\n")
# str1 = Path(allVideos[55])
# str2 = Path(labeledVideos[0])

# # str1 = Path("GHmls16-263_OK/DVD-4/VIDEOS/TRECHO RISER/20161106091420250@DVR-SPARE_Ch1.wmv")
# # str2 = Path("GHmls16-263_OK/DVD-4/20161106091420250@DVR-SPARE_Ch1.wmv")
# #str1 = Path("CIMRL10-676_OK/PIDF-1 PO MRL-021_parte3.mpg")
# #str2 = Path("CIMRL10-676_OK/PIDF-1 PO MRL-021_parte3.mpg")
# print(str1)
# print(str2)
# pattern = ""
# numParts = len(str2.parts)
# for i in range(numParts-1):
#     pattern += str(str2.parts[i])+".*"
# pattern += str(str2.parts[-1])#.replace('.', '\.')
# pattern = str(pattern)
# print("\n", pattern)
# print("len1: ", len(str(str1)))
# print("len2: ", len(str(str2)))

# print("str1==str2: ", str(str1)==str(str2))
# if re.search(pattern, str(str1)):
#     print('regex: True')
# else:
#     print('regex: None')

# #for i in range(len(str(str1))):
#	print("str1: ", str(str1)[i])
#	print("str2: ", str(str2)[i])
#	print("equal? ", str(str1)[i] == str(str2)[i])
