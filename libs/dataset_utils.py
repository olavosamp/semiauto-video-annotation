import os
import math
import warnings
import torch
import torch.nn             as nn
import numpy                as np
import pandas               as pd
import shutil               as sh
import matplotlib.pyplot    as plt
import torch.optim          as optim
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


# Automatic labeling
def automatic_labeling(outputs, outputs_index, upper_thresh, lower_thresh, verbose=True):
    '''
        Return the indexes whose corresponding outputs lie between the given upper and lower thresholds.
    '''
    datasetLen      = len(outputs)
    # indexes         = np.arange(datasetLen, dtype=int)
    indexes         = outputs_index
    upperIndexes    = indexes[np.greater(outputs, upper_thresh)]
    lowerIndexes    = indexes[np.less(outputs, lower_thresh)]
    totalClassified = len(upperIndexes) + len(lowerIndexes)
    
    if verbose:
        print("\nIdeal Upper Threshold: ", upper_thresh)
        print("Ideal Lower Threshold: ", lower_thresh)

        print("\nImages in:")
        print("upperIndexes: ", len(upperIndexes))
        print("lowerIndexes: ", len(lowerIndexes))
        print("\nImages automatically labeled: {}/{} = {:.2f} %".format(totalClassified, datasetLen,
                                                                    (totalClassified)/datasetLen*100))
    return upperIndexes, lowerIndexes


def get_classified_index(index, pos_hashes, neg_hashes, index_col="FrameHash", verbose=True):
    if index_col is not None:
        index.set_index("FrameHash", drop=False, inplace=True)

    positiveLabel = commons.rede1_positive
    negativeLabel = commons.rede1_negative

    newPositives = index.reindex(labels=pos_hashes, axis=0, copy=True)
    newNegatives = index.reindex(labels=neg_hashes, axis=0, copy=True)

    datasetLen    = len(index)
    lenPositives = len(newPositives)
    lenNegatives = len(newNegatives)

    # Set positive and negative class labels
    newPositives["rede1"] = [positiveLabel]*lenPositives
    newNegatives["rede1"] = [negativeLabel]*lenNegatives

    newLabeledIndex = pd.concat([newPositives, newNegatives], axis=0, sort=False)
    if verbose:
        print(newLabeledIndex.shape)
        print("Unlabeled images: ", datasetLen)
        print("New pos labels:   ", lenPositives)
        print("New neg labels:   ", lenNegatives)
        print("Total new labels: ", lenPositives+lenNegatives)
        print("New labels len:   ", newLabeledIndex.shape)
        print("\nAutomatic anotation of {:.2f} % of input images.".format(len(newLabeledIndex)/datasetLen*100))
    return newLabeledIndex


## Threshold finding
def compute_thresholds(val_outputs, labels,
                        upper_ratio=0.95,
                        lower_ratio=0.01,
                        resolution=0.001,
                        val_indexes=None):
    val_outputs = np.squeeze(utils.normalize_array(val_outputs))
    val_outputs = val_outputs[:, 0]
    resBits = len(str(resolution)) -2

    # Maximum resolution is to test a threshold on all output values
    if resolution == 'max':
        upperThreshList = np.sort(val_outputs)
        # upperThreshList = copy(upperThreshList).reverse()
        lowerThreshList = copy(upperThreshList)[::-1]
    else:
        lowerThreshList = np.arange(0., 1., resolution)
        upperThreshList = np.arange(1., 0., -resolution)
    #     upperThreshList = np.arange(0., 1., resolution)
    #     lowerThreshList = np.arange(1., 0., -resolution)

    # Find upper threshold
    # upperThreshList = np.arange(1., 0., -resolution)
    idealUpperThresh = find_ideal_upper_thresh(
                                    val_outputs, labels, upperThreshList, ratio=upper_ratio)#, verbose=True)

    # Find lower threshold
    # lowerThreshList = np.arange(0., 1., resolution)
    idealLowerThresh = find_ideal_lower_thresh(
                                    val_outputs, labels, lowerThreshList, ratio=lower_ratio)

    idealLowerThresh = np.around(idealLowerThresh, decimals=resBits)
    idealUpperThresh = np.around(idealUpperThresh, decimals=resBits)

    ## If thresholds break, take the mean value
    ## TODO: Instead of choosing the mean, choose a thresh that maximizes AUC
    # if idealUpperThresh < idealLowerThresh:
    #     meanThresh = (idealUpperThresh+idealLowerThresh)/2
    #     idealUpperThresh = meanThresh
    #     idealLowerThresh = meanThresh

    if val_indexes is not None:
        automatic_labeling(val_outputs, val_indexes, idealUpperThresh, idealLowerThresh)

    return idealUpperThresh, idealLowerThresh


def upper_positive_relative_ratio(outputs, labels, threshold):
    '''
        Compute ratio of ground truth positive examples above given threshold relative only
        to the examples above the threshold.
    '''
    datasetLen = len(outputs)
    mask       = np.greater(outputs, threshold)
    indexes    = np.arange(datasetLen)[mask]
    
    if len(indexes) > 0:
        posPercent = np.sum(labels[indexes] == 0)/len(indexes) # Positive class index is 0
    else:
        return 1.
    return posPercent


def lower_positive_ratio(outputs, labels, threshold):
    '''
        Compute ratio of ground truth positive examples below a given threshold relative
        to the entire dataset.
    '''
    datasetLen = len(outputs)
    mask       = np.less(outputs, threshold)
    indexes    = np.arange(datasetLen)[mask]

    if len(indexes) > 0:
        posPercent = np.sum(labels[indexes] == 0)/datasetLen # Positive class index is 0
    else:
        return 0.
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

        if (posRatio > ratio) and (ratio > 0.):
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

        if (posRatio < ratio) and (ratio < 1.):
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
        Show Pillow or Pyplot input image.
    '''
    print("Title: ", title_string)
    plt.imshow(image)
    if title_string:
        plt.title(title_string)
    plt.show()


## Dataset files manipulation
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


def index_complement(reference_df, to_drop_df, column_label):
    '''
        Drop rows from 'reference_df' DataFrame indicated by column_label
        column of 'to_drop_df' DataFrame.

        The operation performed can be interpreted a set complement between reference and
        to_drop DataFrames. Returns a DataFrame with length equal to (len(reference_df) - len(to_drop_df)).
    '''
    reference_df.set_index(column_label, drop=False, inplace=True)
    reference_df.drop(labels=to_drop_df[column_label], axis=0, inplace=True)

    reference_df.reset_index(drop=True, inplace=True)
    return reference_df.copy()


def load_outputs_df(outputPath, remove_duplicates=False, softmax=True):
    '''
        Load a pickled dictionary containing a set of outputs, image hashes and labels.
        Each 3-uple corresponds to data of a single sample.
    '''
    pickleData = utils.load_pickle(outputPath)
    
    if remove_duplicates:
        pickleData  = remove_duplicates(pickleData, "ImgHashes")

    outputs      = np.stack(pickleData["Outputs"])
    imgHashes    = pickleData["ImgHashes"]
    labels       = pickleData["Labels"]

    if softmax:  # TODO: Test both options
        outputs = nn.Softmax(dim=1)(torch.as_tensor(outputs))
    else:
        outputs = torch.as_tensor(outputs)
    return outputs.numpy(), imgHashes, labels


def move_dataset_to_train(index_path, dataset_folder, path_column="FramePath", verbose=True):
    ''' Move images from dataset folder to sampled images'''
    def _add_folder_and_copy(x): return utils.copy_files(Path(x), imageFolderPath / Path(x).name)
    index = pd.read_csv(index_path)
    
    if verbose:
        print("\nMoving files from dataset folder to sampled images folder...")
    
    successes = np.sum(index[path_column].map(_add_folder_and_copy))
    if verbose:
        print("{}/{} files moved.".format(successes, len(index[path_column])))
    return successes


def fill_index_information(reference_index, to_fill_index, index_column, columns_to_keep):
    reference_index.set_index(index_column, drop=False, inplace=True)
    to_fill_index.set_index(index_column, drop=False, inplace=True)

    complete_index = reference_index.loc[to_fill_index.index, :]
    for col in columns_to_keep:
        complete_index[col] = to_fill_index[col]

    complete_index.reset_index(drop=True, inplace=True)
    reference_index.reset_index(drop=True, inplace=True)
    to_fill_index.reset_index(drop=True, inplace=True)
    return complete_index.copy()


def merge_manual_auto_sets(auto_df, manual_df):
    manual_df["Annotation"] = [commons.manual_annotation]*len(manual_df)
    auto_df["Annotation"]   = [commons.auto_annotation]*len(auto_df)

    mergedIndex = pd.concat([manual_df, auto_df], axis=0, sort=False)
    return mergedIndex.copy()


def check_df_files(df, check_function, filepath_column, verbose=False):
    '''
        Given a DataFrame with a column of file paths, specified by filepath_column, check if 
        each file passes a True/False check defined by check_function. Return the df without
        the rows that returned False on their checks.
    '''
    dfLen = len(df)
    df.set_index(filepath_column, drop=False, inplace=True)
    labelsToKeep = df[filepath_column].map(check_function)
    df = df[labelsToKeep]
    df.reset_index(drop=True, inplace=True)

    if verbose:
        badLabels = dfLen - len(labelsToKeep)
        print("\nFound and removed {} missing or corrupt entries in the DataFrame.".format(badLabels))

    return df.copy()


def remove_duplicates(target_df, index_column, verbose=False):
    '''
        Drop duplicated entries from target_df DataFrame. Entries are dropped if they have 
        duplicated values in index_column column.
    '''
    target_df.set_index(index_column, drop=False, inplace=True)
    
    numDups = np.sum(target_df.index.duplicated())
    
    target_df = target_df[~target_df.index.duplicated()]
    target_df.reset_index(drop=True, inplace=True)

    if verbose:
        print("\nFound and removed {} duplicated entries in the DataFrame.".format(numDups))

    return target_df.copy()


def move_to_class_folders(indexPath, imageFolderPath, target_net="rede1", verbose=True):
    '''
        Sort labeled images in class folders according to index file with labels and filepaths.
    '''
    indexPath       = Path(indexPath)
    assert imageFolderPath.is_dir(), "Folder argument must be a valid image folder."

    imageIndex = pd.read_csv(indexPath)
    numImages     = len(imageIndex)

    # Get unique tags and create the respective folders
    tags = set(imageIndex[target_net])# - set("-")
    # tags = set([commons.net_classes_table[target_net][x] for x in unprocessedTags])
    print("Found tags ", tags)
    for tag in tags:
        tag = translate_labels(tag, target_net)
        dirs.create_folder(imageFolderPath / tag)
        if verbose:
            print("Created folder ", (imageFolderPath / tag))

    destList = []
    print("Moving files to class folders...")
    for i in tqdm(range(numImages)):
        imageName  = imageIndex.loc[i, 'FrameName']
        # print(imageFolderPath)
        # print(imageName)
        source   = imageFolderPath / imageName
        
        tag      = translate_labels(imageIndex.loc[i, target_net], target_net)
        destName = Path(tag) / imageName
        dest     = imageFolderPath / destName

        # imageIndex.loc[i, 'imagem'] = destName # Unnecessary and possibly harmful
        # print("Moving\n{}\nto\n{}".format(source, dest))
        # input()
        sh.move(source, dest)
        destList.append(dest)
    imageIndex["TrainPath"] = destList
    return imageIndex


def data_folder_split(datasetPath, split_percentages, index=None, seed=None):
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
    assert np.sum(split_percentages) <= 1.0, "Percentages must sum to less or equal to 100%."
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
        # dirs.create_folder(source.parent)
        dirs.create_folder(dest.parent)
        
        sh.move(source, dest)
    
    print("Set lengths:\n\ttrain: {}\n\tval: {}".format(setLengths[0], setLengths[1]))
    print(setLengths)
    print("Moved files to train and val folders in ", datasetPath)

    # Remove old files
    fileList.extend([x.parent for x in fileList])
    fileList = set(fileList)
    print("\nDeleting temporary files...")
    for f in tqdm(fileList):
        # print(f)
        if Path(f).is_dir():
            sh.rmtree(f)
        elif Path(f).is_file():
            os.remove(f)
    
    if index is not None: # Update frame paths in index
        print("\nSaving to index...")
        # def get_name(x): return str(x.name)
        # def add_folder(x): return (Path(dirs.images) / "all_datasets_1s") / x
        def get_parts(x): return "/".join(x.parts[-3:])

        # trainSourceList = list(map(get_name, trainSourceList))
        # valSourceList   = list(map(get_name, valSourceList))
        trainHashList   = utils.make_file_hash_list(trainDestList, hash_column="FrameHash")["FrameHash"]
        valHashList     = utils.make_file_hash_list(valDestList, hash_column="FrameHash")["FrameHash"]

        trainDestList   = list(map(get_parts, trainDestList))
        valDestList     = list(map(get_parts, valDestList))

        index.set_index('FrameHash', drop=False, inplace=True)
        # TODO: Check if train dest paths are guaranteed to be saved in
        #  the same order as index. If not, find other way
        trainIndex              = index.reindex(labels=trainHashList, axis=0, copy=True)
        trainIndex['TrainPath'] = trainDestList
        trainIndex['set']       = ['train']*setLengths[0]

        valIndex                = index.reindex(labels=valHashList, axis=0, copy=True)
        valIndex['TrainPath']   = valDestList
        valIndex['set']         = ['val']*setLengths[1]

        ## This way is unfeasibly slow
        # for i in tqdm(range(setLengths[0])):
        #     index.loc[trainSourceList[i], 'TrainPath'] = trainDestList[i]
        #     index.loc[trainSourceList[i], 'set']    = 'train'
        # for i in tqdm(range(setLengths[1])):
        #     index.loc[valSourceList[i], 'TrainPath']   = valDestList[i]
        #     index.loc[valSourceList[i], 'set']      = 'val'
        index = pd.concat([trainIndex, valIndex], axis=0, sort=False)

        index.reset_index(drop=True, inplace=True)
    return index


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
        # TODO: Replace with right function call
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
    labelsDf[DEF_frameHashColumnName] = utils.make_file_hash_list(labelsDf['FramePath'])['HashMD5']

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


def translate_labels(labels, target_net):
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
        if translatedLabel: # This table formats class labels as task-relevant labels
            return commons.net_classes_table[target_net][translatedLabel]
        else:
            warnings.warn("\nTranslation not found for label:\n\t{}".format(label))
            return commons.no_translation

    # This table formats and normalizes manually annotated class labels
    # Fixes a limited number of common spelling mistakes
    translationTable = commons.classes

    if isinstance(labels, str):
        # If not list, just translate input
        translation = _translate(labels)
    elif hasattr(labels, "__iter__"):
        # If list, apply translation subroutine to every element in list
        translation = list(map(_translate, labels))
    
    return translation
