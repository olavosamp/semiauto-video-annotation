import re
import numpy        as np
import pandas       as pd
from pathlib        import Path
from glob           import glob

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager

def string_list_complement(list1, list2):
    '''
        Arguments:
            list1, list2: Two lists of strings.
        Return value:
            list3: Set complement of the arguments, list1 - list2. Contains elements of list1 that are not in list2.
    '''
    def _compare(path1, path2):
        '''
            Returns True if path1 contains path2, else returns False.
        '''
        pattern = ""
        numParts = len(path2.parts)
        for i in range(numParts-1):
            pattern += str(path2.parts[i])+".*"
        pattern += path2.parts[-1]#.replace('.', '\.')
        pattern = str(pattern)
        if re.search(pattern, str(path1)):
            return True
        else:
            return False

    list3 = []
    for elem1 in list1:
        # print("Searching for\n{}\n".format(elem1))
        # input()
        appendFlag = False
        for elem2 in list2:
            # print("{}\n{}\n{}\n".format(elem1, elem2, _compare(elem1, elem2)))
            if _compare(elem1, elem2):
                # print("Labeled video found. Not adding to list.\n")
                appendFlag = True
                break

        if not(appendFlag):
            list3.append(elem1)
            # print("Labeled video not found for\n{}. Adding to list.\n".format(elem1))
            # print("List size: {}.\n".format(len(list3)))
            # input()

    return list3

def add_ok(pathList):
    '''
        Appends "_OK" to reports created without this termination.

        pathList: List of string paths.
    '''
    def _replace(x):
        for report in commons.reportList:
            x = str(x).replace(report+"/", report+"_OK"+"/")    # Must guarantee to only append _OK to strings without it
            x = str(x).replace(report+"\\", report+"_OK"+"\\")  # Do it twice for Linux/Windows compatibility
        return x
    return list(map(_replace, pathList))


datasetPath = dirs.base_videos

f = lambda x: Path(str(x).strip())
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

# Get video paths in dataset folder (all videos)
allVideos = []
for format in commons.videoFormats:
    globList = glob(datasetPath + "/**" + "/*."+format, recursive=True)
    allVideos.extend(globList)

allVideos = list(map(f, allVideos))
allVideos = list(map(h, allVideos))

# Delete DVD headers
mask = list(map(lambda x: not(x.match("VIDEO_TS.VOB")), allVideos))
allVideos = np.array(allVideos)[mask]


# Get video paths that are unlabeled (all - labeled)
# unlabeledVideos = [x for x in allVideos if x not in labeledVideos]
# unlabeledVideos = set(allVideos) - set(labeledVideos)
unlabeledVideos = string_list_complement(allVideos, labeledVideos)
unlabeledVideos = []
for video in unlabeledVideos:
    print(video)
print("Unlabeled videos: {}, should be {}.".format(len(unlabeledVideos), len(allVideos) - len(labeledVideos)))
print("\n")
for video in allVideos:
    print(video)
print("Total videos: ", len(allVideos))
print("\n")

print("\n")
# str1 = Path(allVideos[48])
# str2 = Path(labeledVideos[0])

# str1 = Path("GHmls16-263_OK/DVD-4/VIDEOS/TRECHO RISER/20161106091420250@DVR-SPARE_Ch1.wmv")
# str2 = Path("GHmls16-263_OK/DVD-4/20161106091420250@DVR-SPARE_Ch1.wmv")
str1 = Path("CIMRL10-676_OK\\PIDF-1 PO MRL-021_parte3.mpg")
str2 = Path("CIMRL10-676_OK\\PIDF-1 PO MRL-021_parte3.mpg")

print(str1)
print(str2)
pattern = ""
numParts = len(str2.parts)
for i in range(numParts-1):
    pattern += str(str2.parts[i])+".*"
pattern += str2.parts[-1].replace('.', '\.')
pattern = str(pattern)

print(pattern)
if re.search(pattern, str(str1)):
    print('True')
else:
    print('None')
