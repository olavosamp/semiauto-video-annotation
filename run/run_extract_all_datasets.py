import numpy                as np
import pandas               as pd
from pathlib                import Path
from glob                   import glob
from datetime               import datetime

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import IndexManager
from libs.get_frames_class  import GetFramesFull

## Dataset settings
datasetName = "all_datasets_1s"
# Local dataset
# datasetPath = dirs.base_videos
# destPath    = dirs.images+datasetName
# Remote dataset
datasetPath = dirs.febe_base_videos
destPath    = dirs.febe_images+datasetName

## Log settings
logWidth1 = 120
logWidth2 = 10
dateStart = datetime.now()
logName = "log "+datasetName +"_{}-{}-{}_{}-{}-{}".format(dateStart.year, dateStart.month,\
 dateStart.day, dateStart.hour, dateStart.minute, dateStart.second)
logPath = Path(dirs.root) / "index" / (logName+".txt")

# Auxiliary lambda functions
f = lambda x: Path(x)
h = lambda x: x.relative_to(datasetPath)

newIndexPath  = Path(dirs.root) / "index" / "unlabeled_index.csv"

ind2  = IndexManager(path=newIndexPath)

# Get video paths in dataset folder (all videos)
# Also search for upper case formats for Linux compatibility
formats = commons.videoFormats
formats.extend([x.upper() for x in commons.videoFormats])

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

# allVideos = allVideos[:2] # Test run with 2 videos
numVideos = len(allVideos)

# Print video paths for checking
for videoPath in allVideos:
    print(videoPath)
print("Total videos: ", len(allVideos))
print("\n")
# exit()

# Create log file
with open(logPath, mode='w') as log:
    log.write("Processing {} videos. Started in {}.".format(numVideos, dateStart)+ "\n")
    log.write("Video".ljust(logWidth1)+ "Frames".ljust(logWidth2)+ "\n")

frameEntryList = []
for i in range(numVideos):
    videoPath = allVideos[i]
    print("\nProcessing video {}/{}".format(i+1, numVideos))
    gff = GetFramesFull(videoPath, videoFolder=datasetPath, destPath=destPath, interval=1, verbose=False)
    newEntries = gff.get_frames()
    frameEntryList.extend(newEntries)

    # Append log
    with open(logPath, mode='a') as log:
        log.write(str(videoPath).ljust(logWidth1)+ str(len(newEntries)).rjust(logWidth2)+ "\n")

dateEnd = datetime.now()

numFrames = len(frameEntryList)
# for i in range(numFrames):
#     entry = frameEntryList[i]
#     print("Adding frame {}/{}".format(i, numFrames))
#     ind2.add_entry(entry)
ind2.add_entry(frameEntryList)

with open(logPath, mode='a') as log:
    log.write("Extraction finished on {}.\nElapsed time {}.\n".format(dateEnd, dateEnd-dateStart))

ind2.write_index(prompt=False)

input("\nExtraction finished. Press any key to end.")
