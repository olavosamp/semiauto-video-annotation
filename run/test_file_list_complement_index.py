import hashlib
import numpy        as np
import pandas       as pd
from pathlib        import Path
from glob           import glob

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager
from libs.utils     import (add_ok,
                            string_list_complement,
                            get_file_list,
                            remove_video_ts,
                            get_relative_list,
                            replace_backslashes)


# Base:    8 VOB + 31 wmv = 39 non-header videos, 40 total
# Compare: 6 VOB + 24 wmv = 30 non-header videos, 31 total
# Dif:     2 VOB + 7 wmv  =  9 non-header videos
baseIndexPath    = Path(dirs.test_assets + "video_list_complement_index/" + "unannotated_index.csv")
compareIndexPath = Path(dirs.test_assets + "video_list_complement_index/" + "annotated_index.csv")

# Get file lists based on folder paths
print("Reading csv...")
baseFileDf    = pd.read_csv(baseIndexPath, dtype=str)
compareFileDf = pd.read_csv(compareIndexPath, dtype=str)

baseFileList    = baseFileDf.loc[:, "VideoPath"]
compareFileList = compareFileDf.loc[:, "VideoPath"]

# Make paths relative to source folders
print("Computing relative paths...")
baseRelFolder   = "/home/common/flexiveis/videos/"
baseFileList    = get_relative_list(baseFileList, baseRelFolder)


# Remove duplicates
print("Calculating unique values...")
baseFileList    = list(dict.fromkeys(baseFileList))
compareFileList = list(dict.fromkeys(compareFileList))

# Add _OK to reports
compareFileList = add_ok(compareFileList)

# Remove VIDEO_TS headers
baseFileList    = remove_video_ts(baseFileList)
compareFileList = remove_video_ts(compareFileList)

# Replace backslashes again, just to be sure
baseFileList    = replace_backslashes(baseFileList)
compareFileList = replace_backslashes(compareFileList)

print("\nBase:\n")
for entry in baseFileList:
    print(entry)

print("\nCompare:\n")
for entry in compareFileList:
    print(entry)

print("\nBase len:\n")
print(len(baseFileList))

print("\nCompare len:\n")
print(len(compareFileList))

# Get list differences
print("")
print("Base: ", len(baseFileList))
print("Compare: ", len(compareFileList))
print("Diff Base - Compare: 94")

print("\nSet difference:")
print(len(set(baseFileList) - set(compareFileList)))

print("\nString list complement:")
print(len(string_list_complement(list(baseFileList), list(compareFileList))))
