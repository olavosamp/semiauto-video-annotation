import pandas       as pd
from pathlib        import Path
from glob           import glob

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager

datasetPath = dirs.base_videos

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
allVideos = list(map(h, allVideos))
for video in allVideos:
    print(video)
print("Total videos: ", len(allVideos))
print("\n")
