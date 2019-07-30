import hashlib
import numpy        as np
import pandas       as pd
from pathlib        import Path
from glob           import glob

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager
from libs.utils     import (string_list_complement,
                            get_file_list,
                            remove_video_ts,
                            get_relative_list)


# Base:    8 VOB + 31 wmv = 39 non-header videos, 40 total
# Compare: 6 VOB + 24 wmv = 30 non-header videos, 31 total
# Dif:     2 VOB + 7 wmv  =  9 non-header videos
baseFolderPath    = Path(dirs.test_assets + "video_list_complement/" + "base_folder/")
compareFolderPath = Path(dirs.test_assets + "video_list_complement/" + "compare_folder/")

# Get file lists based on folder paths
baseFileList    = get_file_list(baseFolderPath, ext_list=commons.videoFormats)
compareFileList = get_file_list(compareFolderPath, ext_list=commons.videoFormats)

# Make paths relative to source folders
baseFileList    = get_relative_list(baseFileList, baseFolderPath)
compareFileList = get_relative_list(compareFileList, compareFolderPath)

# Remove VIDEO_TS headers
baseFileList    = remove_video_ts(baseFileList)
compareFileList = remove_video_ts(compareFileList)

# print(baseFileList)
# print(compareFileList)

print("")
print("Base: ", len(baseFileList))
print("Compare: ", len(compareFileList))

print("\nSet difference:")
print(len(set(baseFileList) - set(compareFileList)))

print("\nString list complement:")
print(len(string_list_complement(list(baseFileList), list(compareFileList))))
