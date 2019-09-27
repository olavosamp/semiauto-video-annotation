import os
import math
import warnings
import torch
import random
import numpy                as np
import pandas               as pd
import shutil               as sh
import matplotlib.pyplot    as plt
from PIL                    import Image
from copy                   import copy
from tqdm                   import tqdm
from glob                   import glob
from pathlib                import Path
from datetime               import datetime

import libs.dirs            as dirs
import libs.commons         as commons
import libs.utils           as utils
from libs.index             import IndexManager
from libs.get_frames_class  import GetFramesFull


## Threshold finding
def compute_thresholds(val_outputs, labels, upper_ratio=0.95, lower_ratio=0.01, resolution=0.001, verbose=True):
    val_outputs = np.squeeze(utils.normalize_array(val_outputs))
    val_outputs = val_outputs[:, 0]
    resBits = len(str(resolution)) -2

    # Find upper threshold
    upperThreshList = np.arange(1., 0., -resolution)
    idealUpperThresh = find_ideal_upper_thresh(
                                    val_outputs, labels, upperThreshList, ratio=upper_ratio)#, verbose=True)

    # Find lower threshold
    lowerThreshList = np.arange(0., 1., resolution)
    idealLowerThresh = find_ideal_lower_thresh(
                                    val_outputs, labels, lowerThreshList, ratio=lower_ratio)

    idealLowerThresh = np.around(idealLowerThresh, decimals=resBits)
    idealUpperThresh = np.around(idealUpperThresh, decimals=resBits)
    if verbose:
        automatic_labeling(val_outputs, idealUpperThresh, idealLowerThresh)

    return idealUpperThresh, idealLowerThresh


def automatic_labeling(outputs, upper_thresh, lower_thresh):
    datasetLen      = len(outputs)
    indexes         = np.arange(datasetLen, dtype=int)
    upperIndexes    = indexes[np.greater(outputs, upper_thresh)]
    lowerIndexes    = indexes[np.less(outputs, lower_thresh)]
    totalClassified = len(upperIndexes) + len(lowerIndexes)

    print("\nIdeal Upper Threshold: ", upper_thresh)
    print("Ideal Lower Threshold: ", lower_thresh)

    print("\nResults in Validation set:")
    print("upperIndexes: ", len(upperIndexes))
    print("lowerIndexes: ", len(lowerIndexes))
    print("\nImages automatically labeled: {}/{} = {:.2f} %".format(totalClassified, datasetLen,
                                                                (totalClassified)/datasetLen*100))
    return upperIndexes, lowerIndexes


def upper_positive_relative_ratio(outputs, labels, threshold):
    '''
        Compute ratio of ground truth positive examples above given threshold relative only
        to the examples above the threshold.
    '''
    datasetLen     = len(outputs)
    mask      = np.greater(outputs, threshold)
    indexes   = np.arange(datasetLen)[mask]
    
    posPercent = np.sum(labels[indexes] == 0)/len(indexes) # Positive class index is 0
    return posPercent


def lower_positive_ratio(outputs, labels, threshold):
    '''
        Compute ratio of ground truth positive examples below a given threshold relative
        to the entire dataset.
    '''
    datasetLen = len(outputs)
    mask       = np.less(outputs, threshold)
    indexes    = np.arange(datasetLen)[mask]

    posPercent = np.sum(labels[indexes] == 0)/datasetLen # Positive class index is 0
    return posPercent


def find_ideal_lower_thresh(outputs, labels, threshold_list=None, ratio=0.01, resolution=0.001, verbose=False):
    if verbose:
        print("\nThreshold\tLower Pos Ratio")
    
    if threshold_list is None:
        threshold_list = np.arange(0., 1., resolution)
    
    for i in tqdm(range(len(threshold_list))):
        lowerThresh = threshold_list[i]
        posRatio = lower_positive_ratio(outputs, labels, lowerThresh)

        if verbose:
            print("{:.2f}\t\t{:.2f}".format(lowerThresh, posRatio)) # Print search progress

        if posRatio > ratio:
            if i-1 < 0:
                print("\nThreshold could not be found.")
                return None
            idealThresh = threshold_list[i-1]
            posRatio = lower_positive_ratio(outputs, labels, idealThresh)

            print("\nFound ideal Lower threshold {:.3f} with {:.2f} % ground truth positives.".format(idealThresh, posRatio*100))
            return idealThresh


def find_ideal_upper_thresh(outputs, labels, threshold_list=None, ratio=0.95, resolution=0.001, verbose=False):
    if verbose:
        print("\nThreshold\tUpper Pos Ratio")
    
    if threshold_list is None:
        threshold_list = np.arange(1., 0., -resolution)
    
    for i in tqdm(range(len(threshold_list))):
        upperThresh = threshold_list[i]
        posRatio = upper_positive_relative_ratio(outputs, labels, upperThresh)

        if verbose:
            print("{:.2f}\t\t{:.2f}".format(upperThresh, posRatio)) # Print search progress

        if posRatio < ratio:
            if i-1 < 0:
                print("\nThreshold could not be found.")
                return None
            idealThresh = threshold_list[i-1]
            posRatio = upper_positive_relative_ratio(outputs, labels, idealThresh)

            print("\nFound ideal Upper threshold {:.3f} with {:.2f} % ground truth positives.".format(idealThresh, posRatio*100))
            return idealThresh


## Image processing
def show_inputs(inputs, labels):
    '''
        Function to visualize dataset inputs
    '''
    for i in range(len(inputs)):
        print(np.shape(inputs.cpu().numpy()[i,:,:,:]))
        img = np.transpose(inputs.cpu().numpy()[i,:,:,:], (1, 2, 0))
        print(np.shape(img))
        print(labels.size())
        print("Label: ", labels[i])
        plt.imshow(img)
        plt.title("Label: {}".format(labels[i]))
        plt.show()


def show_image(image, title_string=None):
    '''
        Show input image.
    '''
    print("Title: ", title_string)
    plt.imshow(image)
    if title_string:
        plt.title(title_string)
    plt.show()


## Pytorch utilities
def set_torch_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexLoader:
    '''
        Iterator to load and transform an image and its file hash.
        
        Construtctor arguments:

            imagePathList: list of strings

            label_list: list of ints

            transform: Torchvision transform


        Returns img, imgHash, label (optional)
        
            img: Tensor of a Pillow Image
                Torch tensor of a Pillow Image. The input transforms are applied and it has shape (channels, h, w)
            imgHash: string
                MD5 hash of input image.
            label: int
                Numeric class label associated with img. Will only be returned if InputLoader received label_list as input.
    '''
    def __init__(self, imagePathList, label_list=None, batch_size=4, transform=None):
        self.imagePathList  = imagePathList
        self.batch_size     = batch_size
        self.transform      = transform
        self.label_list     = label_list
        
        self.current_index  = 0
        self.datasetLen     = len(self.imagePathList)

        if self.label_list is not None:
            assert len(self.label_list) == self.datasetLen, "Image path and label lists must be of same size."

        # TODO: (maybe) add default Compose transform with ToTensor
        # and Transpose to return a Tensor image with shape (channel, width, height)
        # if self.transform != None:
        #     self.transform = transforms.ToTensor()

    def __len__(self):
        return math.ceil((self.datasetLen - self.current_index) / self.batch_size)

    def __iter__(self):
        return self
    
    def __next__(self):
        while self.current_index+self.batch_size > self.datasetLen:
            self.batch_size -= 1
            if self.batch_size == 0:
                raise StopIteration

        imgList     = [] 
        imgHashList = []
        labelList   = []
        for _ in range(self.batch_size):
            imgHash = utils.file_hash(self.imagePathList[self.current_index])
            img = Image.open(self.imagePathList[self.current_index])

            if self.transform:
                img = self.transform(img)

            if self.label_list is not None:
                label = self.label_list[self.current_index]
                labelList.append(label)
            
            imgList.append(img)
            imgHashList.append(imgHash)

            self.current_index += 1
        
        imgList = torch.stack(imgList, dim=0)

        if self.label_list is None:
            return imgList, imgHashList
        else:
            return imgList, imgHashList, labelList


def move_to_class_folders(indexPath, imageFolder="sampled_images"):
    iterationFolder = Path(indexPath).parent
    imageFolder     = iterationFolder / imageFolder
    assert imageFolder.is_dir(), "Folder argument must be a valid image folder."

    targetNet = 'rede2'

    imageIndex = pd.read_csv(indexPath)
    numImages  = len(imageIndex)

    # Get unique tags and create the respective folders
    tags = set(imageIndex[targetNet])# - set("-")
    for tag in tags:
        tag = translate_labels(tag)
        dirs.create_folder(imageFolder / tag)

    print("Moving files to class folders...")
    for i in tqdm(range(numImages)):
        imgName = imageIndex.loc[i, 'imagem']
        source  = imageFolder / imgName
        dest    = imageFolder / translate_labels(imageIndex.loc[i, targetNet]) / imgName

        sh.move(source, dest)


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
    assert np.sum(split_percentages) <= 1.0, "Percentages must sum to less than 100%."
    datasetPath = Path(datasetPath)

    # Get file list as Path objects
    fileList   = utils.make_path(utils.get_file_list(str(datasetPath), ext_list=['jpg', 'png']))

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

    print("Moving files to set folders...")
    for source, dest in tqdm(zip(sources, dests)):
        dirs.create_folder(source.parent)
        dirs.create_folder(dest.parent)
        
        sh.move(source, dest)
    
    print("Set lengths:\n\ttrain: {}\n\tval: {}".format(setLengths[0], setLengths[1]))
    print(setLengths)
    print("Moved files to train and val folders in ", datasetPath)

    # Remove old files
    for f in fileList:
        # print("Deleting ", f)
        if Path(f).is_dir():
            sh.rmtree(f)
        elif Path(f).is_file():
            os.remove(f)


def translate_interface_labels_file(filePath):
    assert utils.file_exists(filePath), "File doesn't exist."
    
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
    framePathList = utils.get_file_list(str(labelsFile.parent), ext_list=['jpg'])
    framePathList = list(map(utils.func_make_path, framePathList))    # Make all filepaths Path objects

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
    labelsDf[DEF_frameHashColumnName] = utils.make_video_hash_list(labelsDf['FramePath'])['HashMD5']

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
        logName = "log_{}_{}".format(datasetName, utils.get_time_string(dateStart))
        logPath = Path(logFolder) / (logName + ".txt")

    if indexPath == 'auto':
        indexPath = dirs.root+"index/unlabeled_index_{}.csv".format(utils.get_time_string(dateStart))
    
    indexPath = Path(indexPath)
    dirs.create_folder(indexPath.parent)

    # Auxiliary lambda functions
    def func_relative_to(x): return x.relative_to(videoFolder)

    index  = IndexManager(path=indexPath)

    # Get video paths in dataset folder (all videos)
    # Also search for upper case formats for Linux compatibility
    # TODO: Replace with a simpler case-insensitive search
    allVideos = utils.get_file_list(videoFolder, ext_list=commons.videoFormats)

    # Make every entry a Path object
    allVideos = list(map(utils.func_make_path, allVideos))
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
        # videoDestFolder = destFolder / videoPath.relative_to(videoFolder)
        
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
            warnings.warn("\nTranslation not found for label:\n\t{}".format(label))
            # print("Translation not found for label:\n\t{}".format(label))
            # input()
            return commons.no_translation

    translationTable = commons.classes
    if isinstance(labels, str):
        # If not list, just translate input
        translation = _translate(labels)
    elif hasattr(labels, "__iter__"):
        # If list, apply translation subroutine to every element in list
        translation = list(map(_translate, labels))
    
    return translation
