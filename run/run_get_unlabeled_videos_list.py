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

f = lambda x: Path(str(x).replace("\\", "/").replace(" ", "_"))
h = lambda x: x.relative_to(datasetPath)

mainIndexPath = Path(dirs.index)
newIndexPath  = Path(dirs.root) / "index" / "unlabeled_index.csv"

ind1  = IndexManager(path=mainIndexPath)            # Existing image index
ind2  = IndexManager(path=newIndexPath)             # New image index

# Get video paths in Main Index (labeled videos)
labeledVideos = list(ind1.index.VideoPath.unique())
labeledVideos = add_ok(labeledVideos)               # Append _OK to reports without it
labeledVideos = list(map(f, labeledVideos))         # Make every element a Path object
labeledVideos = list(dict.fromkeys(labeledVideos))  # Remove duplicated elements

for video in labeledVideos:
    print(video)
print("Labeled videos: ", len(labeledVideos))
print("\n")

#print("debug")

#print(str(labeledVideos[0]).replace("\\", "/"))
#print(str(labeledVideos[0]))
#print(Path(labeledVideos[0]))
#input()

# Get video paths in dataset folder (all videos)
allVideos = []
for format in commons.videoFormats:
    globString = datasetPath + "**" + "/*."+format
    globList = glob(globString, recursive=True)
#    print(globString)
#    input()
    allVideos.extend(globList)

allVideos = list(map(f, allVideos))
allVideos = list(map(h, allVideos))

# Delete DVD headers
mask = list(map(lambda x: not(x.match("VIDEO_TS.VOB")), allVideos))
allVideos = np.array(allVideos)[mask]

for video in allVideos:
    print(video)
print("Total videos: ", len(allVideos))
print("\n")

# Get video paths that are unlabeled (all - labeled)
# unlabeledVideos = [x for x in allVideos if x not in labeledVideos]
# unlabeledVideos = set(allVideos) - set(labeledVideos)
unlabeledVideos = string_list_complement(allVideos, labeledVideos)
for video in unlabeledVideos:
    print(video)
print("Unlabeled videos: {}, should be {}.".format(len(unlabeledVideos), len(allVideos) - len(labeledVideos)))
print("\n")

print("\n")
str1 = Path(allVideos[55])
str2 = Path(labeledVideos[0])

# str1 = Path("GHmls16-263_OK/DVD-4/VIDEOS/TRECHO RISER/20161106091420250@DVR-SPARE_Ch1.wmv")
# str2 = Path("GHmls16-263_OK/DVD-4/20161106091420250@DVR-SPARE_Ch1.wmv")
#str1 = Path("CIMRL10-676_OK/PIDF-1 PO MRL-021_parte3.mpg")
#str2 = Path("CIMRL10-676_OK/PIDF-1 PO MRL-021_parte3.mpg")
print(str1)
print(str2)
pattern = ""
numParts = len(str2.parts)
for i in range(numParts-1):
    pattern += str(str2.parts[i])+".*"
pattern += str(str2.parts[-1])#.replace('.', '\.')
pattern = str(pattern)
print("\n", pattern)
print("len1: ", len(str(str1)))
print("len2: ", len(str(str2)))

print("str1==str2: ", str(str1)==str(str2))
if re.search(pattern, str(str1)):
    print('regex: True')
else:
    print('regex: None')

#for i in range(len(str(str1))):
#	print("str1: ", str(str1)[i])
#	print("str2: ", str(str2)[i])
#	print("equal? ", str(str1)[i] == str(str2)[i])
