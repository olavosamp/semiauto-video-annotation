import numpy                as np
import pandas               as pd
from pathlib                import Path
from glob                   import glob
from datetime               import datetime

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import IndexManager
from libs.get_frames_class  import GetFramesFull
from libs.utils             import *


def extract_dataset(videoFolder, destFolder,
                 datasetName="unlabeled_dataset_test",
                 indexPath='auto',
                 logFolder=dirs.index,
                 verbose=True):
    '''
        Captura todos os frames de todos os vídeos da pasta de vídeos em videoFolder,
        move as imagens para destFolder e registra no índice em newIndexPath
    '''
    ## Log settings
    logWidth1 = 120
    logWidth2 = 10
    dateStart = datetime.now()
    if logFolder:
        # TODO: Fix double date log name bug
        logName = "log_{}_{}".format(datasetName, get_time_string(dateStart))
        logPath = Path(logFolder) / (logName + ".txt")

    if indexPath == 'auto':
        indexPath = dirs.root+"index/unlabeled_index_{}.csv".format(get_time_string(dateStart))
    
    indexPath = Path(indexPath)
    dirs.create_folder(indexPath.parent)

    # Auxiliary lambda functions
    def func_make_path(x):   return Path(x)
    def func_relative_to(x): return x.relative_to(videoFolder)

    index  = IndexManager(path=indexPath)

    # Get video paths in dataset folder (all videos)
    # Also search for upper case formats for Linux compatibility
    # TODO: Replace with a simpler case-insensitive search
    allVideos = get_file_list(videoFolder, ext_list=commons.videoFormats)

    # Make every entry a Path object
    allVideos = list(map(func_make_path, allVideos))
    # allVideos = list(map(h, allVideos))

    # Delete DVD headers
    mask = list(map(lambda x: not(x.match("VIDEO_TS.VOB")), allVideos))
    allVideos = np.array(allVideos)[mask]

    allVideos = allVideos[:3] # Test run with 10 videos
    numVideos = len(allVideos)

    if verbose:
        # Print video paths for checking
        for videoPath in allVideos:
            print(videoPath)
        print("Total videos: ", len(allVideos))
        print("\n")

    if logFolder:
        # Create log file
        with open(logPath, mode='w') as log:
            log.write("Processing {} videos. Started in {}.".format(numVideos, dateStart)+ "\n")
            log.write("Video".ljust(logWidth1)+ "Frames".ljust(logWidth2)+ "\n")

    print("Processing {} videos. Started in {}.".format(numVideos, dateStart)+ "\n")
    frameEntryList = []
    for i in range(numVideos):
        videoPath = allVideos[i]
        videoDestFolder = destFolder / videoPath.relative_to(videoFolder)
        print("\nProcessing video {}/{}".format(i+1, numVideos))
        gff = GetFramesFull(videoPath, videoFolder=videoFolder, destPath=destFolder, interval=1, verbose=False)
        newEntries = gff.get_frames()
        frameEntryList.extend(newEntries)

        if logFolder:
            # Append log
            with open(logPath, mode='a') as log:
                log.write(str(videoPath).ljust(logWidth1)+ str(len(newEntries)).rjust(logWidth2)+ "\n")
        del gff

    dateEnd = datetime.now()

    index.add_entry(frameEntryList)
    
    if logFolder:
        with open(logPath, mode='a') as log:
            log.write("Extraction finished on {}.\nElapsed time {}.\n".format(dateEnd, dateEnd-dateStart))

    index.write_index(prompt=False)

    input("\nExtraction finished. Press any key to end.")

    return index

if __name__ == "__main__":
    ## Dataset settings
    datasetName = "test_all_datasets_1s"

    # Local dataset
    datasetPath   = dirs.base_videos
    destFolder    = dirs.images+datasetName
    # Remote dataset
    # datasetPath   = dirs.febe_base_videos
    # destFolder    = dirs.febe_images+datasetName

    newIndexPath  = Path(dirs.root) / "index" / "test_index.csv"

    unlabeledIndex = extract_dataset(datasetPath, destFolder,
                                    datasetName="unlabeled_dataset_test",
                                    indexPath="auto")

    print("Index shape: ", unlabeledIndex.index.shape)