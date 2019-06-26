import numpy                as np
import pandas               as pd
from pathlib                import Path
from glob                   import glob

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import IndexManager
from libs.get_frames_class  import GetFramesFull

# datasetPath = dirs.base_videos
# destPath    = dirs.images+"/all_datasets_1s/"
datasetPath = dirs.febe_base_videos
destPath    = dirs.febe_images+"/all_datasets_1s/"

f = lambda x: Path(x)
h = lambda x: x.relative_to(datasetPath)

# mainIndexPath = Path(dirs.index) / "index" / "main_index.csv"
newIndexPath  = Path(dirs.root) / "index" / "unlabeled_index.csv"

ind2  = IndexManager(path=newIndexPath)

# Also search for upper case formats for Linux compatibility
formats = commons.videoFormats
formats.extend([x.upper() for x in commons.videoFormats])

# Get video paths in dataset folder (all videos)
allVideos = []
for format in formats:
    globString = datasetPath + "/**" + "/*."+format
    globList = glob(globString, recursive=True)
    allVideos.extend(globList)

# Remove duplicated entries
allVideos = list(dict.fromkeys(allVideos))

# Make every entry a Path object
allVideos = list(map(f, allVideos))
# allVideos = list(map(h, allVideos))

# Delete DVD headers
mask = list(map(lambda x: not(x.match("VIDEO_TS.VOB")), allVideos))
allVideos = np.array(allVideos)[mask]

for videoPath in allVideos:
    print(videoPath)
print("Total videos: ", len(allVideos))
print("\n")
# exit()

allVideos = allVideos[:2]
numVideos = len(allVideos)
frameEntryList = []
for i in range(numVideos):
    videoPath = allVideos[i]
    print("Processing video {}/{}".format(i+1, numVideos))
    gff = GetFramesFull(videoPath, destPath=destPath, interval=1, verbose=False)
    newEntries = gff.get_frames()
    frameEntryList.extend(newEntries)

for entry in frameEntryList:
    ind2.add_entry(entry)

ind2.write_index()
ind2.report_changes()
