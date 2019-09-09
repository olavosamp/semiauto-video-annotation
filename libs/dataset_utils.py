import os
import numpy                as np
import pandas               as pd
import shutil               as sh
from copy                   import copy
from tqdm                   import tqdm
from glob                   import glob
from pathlib                import Path
from datetime               import datetime

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import *
from libs.get_frames_class  import GetFramesFull
from libs.utils             import *


def data_folder_split(datasetPath, split_percentages, seed=None):
    '''
        Split dataset images in train and validation sets. Move image files found
        at datasetPath to two folders: datasetPath/train/ and datasetPath/val/, according
        to the given split percentages.

        datasetPath: filepath string
            Dataset root folder.
        
        split_percentages: list of positive floats
            List of dataset split percentages. Following the order [train, validation, test],
            each number represents the percentage of examples that will be allocated to
            the respective set.

        OBS: Currently only implemented for train and validation sets
    '''
    def _add_set_name(x, name):
        return datasetPath / name / Path(x).relative_to(datasetPath)
    
    if seed:
        np.random.seed(seed)

    assert len(split_percentages) == 2, "List must contain only train and val percentages."
    datasetPath = Path(datasetPath)

    # Get file list as Path objects
    fileList   = make_path(get_file_list(str(datasetPath), ext_list=['jpg', 'png']))

    datasetLen = len(fileList)
    print(datasetPath)
    print(datasetLen)

    # Compute size of each set
    setLengths      = np.zeros(len(split_percentages), dtype=int)
    setLengths[:-1] = np.multiply(split_percentages[:-1], datasetLen).astype(int)
    setLengths[-1]  = (datasetLen - setLengths.sum()) # Last size is the number of remaining examples

    assert setLengths.sum() == datasetLen, "Error: Set sizes doesn't sum to total size."

    # Shuffle list and sample examples for each set
    np.random.shuffle(fileList)
    trainSourceList = fileList[:setLengths[0]]
    valSourceList   = fileList[setLengths[0]:]
    
    trainDestList   = list(map(_add_set_name, trainSourceList, ['train']*int(setLengths[0])))
    valDestList     = list(map(_add_set_name, valSourceList, ['val']*int(setLengths[0])))

    sources = copy(trainSourceList)
    dests   = copy(trainDestList)

    sources.extend(valSourceList)
    dests.extend(valDestList)

    for source, dest in zip(sources, dests):
        dirs.create_folder(source.parent)
        dirs.create_folder(dest.parent)
        
        copy_files(source, dest)
    
    print("Set lengths:\n\ttrain: {}\n\tval: {}".format(setLengths[0], setLengths[1]))
    print(setLengths)
    print("Moved files to train and val folders in ", datasetPath)

    # Remove old files
    for f in fileList:
        # print("Deleting ", f)
        if Path(f).is_dir():
            sh.rmtree(f)
        else:
            os.remove(f)


def translate_interface_labels_file(filePath):
    assert file_exists(filePath), "File doesn't exist."
    
    # Read index
    newLabelsIndex = pd.read_csv(filePath)
    newLabelLen = newLabelsIndex.shape[0]
    
    # Get new tags from index columns
    tagList = []
    for i in range(newLabelLen):
        # Discard duplicated tags
        newTags = list(dict.fromkeys([newLabelsIndex.loc[i, 'rede1'],
                                        newLabelsIndex.loc[i, 'rede2'],
                                        newLabelsIndex.loc[i, 'rede3']]))
        if newTags.count("-") > 0:
            newTags.remove("-")
        tagList.append(translate_labels(newTags))
    
    newLabelsIndex['Tags'] = tagList
    return newLabelsIndex


def add_frame_hash_to_labels_file(labelsFile, framePathColumn='imagem'):
    '''
        Compute MD5 hashes of frames in a interface-generated labels csv file.
        File must be in a parent folder of the labeled images'.

        Adds a column called DEF_frameHashColumnName with file hashes of the images found in
    '''
    DEF_frameHashColumnName = 'FrameHash'

    labelsFile    = Path(labelsFile)
    labelsDf      = pd.read_csv(labelsFile)
    
    # Get filepaths of images in child directories
    framePathList = get_file_list(str(labelsFile.parent), ext_list=['jpg'])
    framePathList = list(map(func_make_path, framePathList))    # Make all filepaths Path objects

    labelsDf.set_index(framePathColumn, drop=False, inplace=True)

    # Compute framePaths
    try:
        for framePath in framePathList:
            labelsDf.loc[framePath.name, 'FramePath'] = framePath
    except KeyError as e:
        print("KeyError in add_frame_hash_to_labels_file: probably a frame\
               path did not have a corresponding framePathColumns correspondent key.\
               Did you move any interface-generated image?\n\n")
        raise KeyError(e)

    # Drop frame name index
    labelsDf.reset_index(drop=True, inplace=True)

    # Compute and add frame hashes
    labelsDf[DEF_frameHashColumnName] = make_video_hash_list(labelsDf['FramePath'])['HashMD5']

    # Drop FramePath column
    labelsDf.drop('FramePath', axis=1, inplace=True)

    labelsDf.to_csv(labelsFile, index=False)
    return labelsDf


def extract_dataset(videoFolder, destFolder,
                    datasetName="unlabeled_dataset_test",
                    indexPath='auto',
                    logFolder=dirs.index,
                    verbose=True):
    '''
        Captura todos os frames de todos os vídeos da pasta de vídeos em videoFolder,
        move as imagens para destFolder e registra no índice em indexPath
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
    def func_relative_to(x): return x.relative_to(videoFolder)

    index  = IndexManager(path=indexPath)

    # Get video paths in dataset folder (all videos)
    # Also search for upper case formats for Linux compatibility
    # TODO: Replace with a simpler case-insensitive search
    allVideos = get_file_list(videoFolder, ext_list=commons.videoFormats)

    # Make every entry a Path object
    allVideos = list(map(func_make_path, allVideos))
    # allVideos = list(map(func_relative_to, allVideos))

    # Delete DVD headers
    mask = list(map(lambda x: not(x.match("VIDEO_TS.VOB")), allVideos))
    allVideos = np.array(allVideos)[mask]

    # allVideos = allVideos[:2] # Test run with x videos
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
            log.write("Processing {} videos. Started in {}.".format(numVideos, dateStart) + "\n")
            log.write("Video".ljust(logWidth1) + "Frames".ljust(logWidth2) + "\n")

    print("Processing {} videos. Started in {}.".format(numVideos, dateStart) + "\n")
    frameEntryList = []
    for i in tqdm(range(numVideos)):
        videoPath = allVideos[i]
        videoDestFolder = destFolder / videoPath.relative_to(videoFolder)
        
        gff = GetFramesFull(videoPath, videoFolder=videoFolder,
                            destPath=destFolder, interval=1, verbose=False)
        newEntries = gff.get_frames()
        frameEntryList.extend(newEntries)

        if logFolder:
            # Append log
            with open(logPath, mode='a') as log:
                log.write(str(videoPath).ljust(logWidth1)+ str(len(newEntries)).rjust(logWidth2)+ "\n")
        del gff

    dateEnd = datetime.now()

    index.add_entry(frameEntryList)
    index.compute_frame_hashes()
    
    if logFolder:
        with open(logPath, mode='a') as log:
            log.write("Extraction finished on {}.\nElapsed time {}.\n".format(dateEnd, dateEnd-dateStart))

    index.write_index(dest_path=indexPath, prompt=False)

    input("\nExtraction finished. Press any key to end.")

    return index


def translate_labels(labels):
    '''
        Translate interface-generated labels to the index standard, following commons.classes
         class list.

        Argument:
            label: string or list of strings
                A string or list of strings representing one or more classes, following
                the pattern of interface-generated labels.

        Returns:
            translation: string or list of strings
                A string or list of strings, representing the translation of each input string,
                according to the dictionary at commons.classes.
    '''
    def _translate(label):
        translatedLabel = None
        for tup in translationTable.items():
            for value in tup[1]:
                if label.lower() == value.lower():
                    translatedLabel = str(tup[0])
        if translatedLabel:
            return translatedLabel
        else:
            raise KeyError("Translation not found for this label.")


    translationTable = commons.classes
    if hasattr(labels, "__len__"):
        # If list, apply translation subroutine to every element in list
        translation = list(map(_translate, labels))
    else:
        # If not list, just translate input
        translation = _translate(labelList)
    # print(translation)
    # input()
    return translation
