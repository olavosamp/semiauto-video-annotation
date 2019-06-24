import pandas       as pd
from pathlib        import Path
from glob           import glob

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager

f = lambda x: Path(x)

mainIndexPath = Path(dirs.index) / "main_index.csv"
newIndexPath  = Path(dirs.index) / "unlabeled_index.csv"

ind1  = IndexManager(path=mainIndexPath)
ind2  = IndexManager(path=newIndexPath)

# Get video paths in Main Index (labeled videos)
labeledVideos = list(ind1.index.VideoPath.unique())
labeledVideos = list(map(f, labeledVideos))
for video in labeledVideos:
    print(video)
print("Labeled videos: ", len(labeledVideos))
print("\n")

# Get video paths in dataset folder (all videos)
datasetPath = dirs.base_videos
allVideos = []
for format in commons.videoFormats:
    globList = glob(datasetPath + "/**" + "/*."+format, recursive=True)
    allVideos.extend(globList)

allVideos = list(map(f, allVideos))
for video in allVideos:
    print(video)
print("Total videos: ", len(allVideos))
print("\n")

# Get video paths that are unlabeled (all - labeled)
# unlabeledVideos = [x for x in allVideos if x not in labeledVideos]
unlabeledVideos = set(allVideos) - set(labeledVideos)

for video in unlabeledVideos:
    print(video)
print("Unlabeled videos: ", len(unlabeledVideos))
print("\n")

print("\n")
print(allVideos[-2])
print(labeledVideos[-1])
print(allVideos[-2] == labeledVideos[-1])
