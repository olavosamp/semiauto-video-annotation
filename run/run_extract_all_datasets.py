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

# ind1  = IndexManager(path=mainIndexPath)
ind2  = IndexManager(path=newIndexPath)

# Get video paths in dataset folder (all videos)
allVideos = []
for format in commons.videoFormats:
    globList = glob(datasetPath + "/**" + "/*."+format, recursive=True)
    allVideos.extend(globList)

allVideos = list(map(f, allVideos))
# allVideos = list(map(h, allVideos))

for videoPath in allVideos:
    print(videoPath)
print("Total videos: ", len(allVideos))
print("\n")

frameEntryList = []
numVideos = len(allVideos)
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
