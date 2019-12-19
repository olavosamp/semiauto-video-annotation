import os
import math
import warnings
import torch
import torch.nn             as nn
import numpy                as np
import pandas               as pd
import shutil               as sh
from glob                   import glob
from PIL                    import Image
from copy                   import copy
from tqdm                   import tqdm
from pathlib                import Path
from datetime               import datetime

import libs.dirs            as dirs
import libs.commons         as commons
import libs.utils           as utils
from libs.index             import IndexManager
from libs.get_frames_class  import GetFramesFull

# User Input
def get_input_target_class(net_class_dict):
    '''
        Get user input of net target class. Applicable to rede3 only.
    '''
    classLen = len(net_class_dict)
    print("Enter the target class code from list:\n")
    print("Code\tClass name")
    for i in range(classLen):
        print("{}:\t{}".format(i, net_class_dict[i]))

    input_class_code = int(input())
    if input_class_code < classLen:
        event_class = net_class_dict[input_class_code]
    else:
        event_class = "UNKNOWN"

    while event_class not in net_class_dict.values():
        input_class_code = input("Unknown class. Please select a class from the list.\n")
        try:
            input_class_code = int(input_class_code)
        except ValueError:
            continue

        if input_class_code < classLen:
            event_class = net_class_dict[input_class_code]
    return event_class


# Reports and logging
def get_class_counts(index, class_column, pos_label, neg_label):
    '''
        Returns index class counts according to input labels.

        index: pandas DataFrame
            DataFrame with the elements to be counted.

        pos_label, neg_label: any object or list
            labels to be compared to elements of index[class_column]. If any neg_label
            is None, the count of negative elements will be <Total index size> - <positive count>.
    '''
    if isinstance(pos_label, str) or not(hasattr(pos_label, "__iter__")):
        pos_label = [pos_label]
    if isinstance(neg_label, str) or not(hasattr(neg_label, "__iter__")):
        neg_label = [neg_label]

    posCount = 0
    for label in pos_label:
        posCount += index.groupby(class_column).get_group(label).count()[0]

    negCount = 0
    for label in neg_label:
        if label is None:
            negCount = index.shape[0] - posCount
            break
        # Else, count normally
        negCount += index.groupby(class_column).get_group(label).count()[0]
    return posCount, negCount


def get_net_class_counts(index_path, net, target_class=None):
    '''
        Chooses correct class labels to use in a get_class_counts function call
        according to input net and target_class.
    '''
    assert Path(index_path).is_file(), "Index path does not exist."
    index = remove_duplicates(pd.read_csv(index_path, low_memory=False), "FrameHash")

    if (net == 3) and (target_class not in commons.rede3_classes.values()):
        raise ValueError("Net 3 requires a valid target_class.")

    if net == 1:
        classColumn = "rede1"
        posLabel = commons.rede1_positive
        negLabel = commons.rede1_negative
        
        mask = None
    elif net ==2:
        classColumn = "rede2"
        posLabel = commons.rede2_positive
        negLabel = commons.rede2_negative

        mask = (index["rede1"] == commons.rede1_positive)
    elif net == 3:
        classColumn = "rede3"
        posLabel = target_class
        negLabel = None

        mask = (index["rede2"] == commons.rede2_positive)
    
    if mask is not None:
        # Pass only relevant fraction of index to get_class_counts
        index = index.loc[mask, :]

    # Translate to binary classes
    index[classColumn] = translate_labels(index[classColumn], classColumn)
    
    return get_class_counts(index, classColumn, posLabel, negLabel)


def save_seed_log(log_path, seed, id_string):
    # Save sample seed
    if Path(log_path).is_file():
        f = open(log_path, 'a')
    else:
        f = open(log_path, 'w')
    f.write("{}\n{}\n".format(id_string, seed))
    f.close()


def get_loop_stats(loop_folder): # TODO: Finish function
    statsDf = pd.DataFrame()
    return statsDf


def make_report(report_path, sampled_path, manual_path, automatic_path, prev_unlabeled_path,
                train_info, rede=1, target_class=None, show=False):
    sampledIndex     = pd.read_csv(sampled_path)
    manualIndex      = pd.read_csv(manual_path)
    autoIndex        = pd.read_csv(automatic_path)
    prevUnlabelIndex = pd.read_csv(prev_unlabeled_path)

    # Get report information
    numUnlabel = prevUnlabelIndex.shape[0]
    numSampled = sampledIndex.shape[0]
    
    sampledNaoDuto = 0
    if rede == 1:
        sampledNaoDuto = sampledIndex.groupby("rede1").get_group("Confuso").count()[0]+\
                         sampledIndex.groupby("rede1").get_group("Nada").count()[0]

    sampledDuto      = sampledIndex.groupby("rede1").get_group(commons.rede1_positive).count()[0]

    sampledNaoEvento = 0
    sampledEvento    = sampledIndex.groupby("rede2").get_group(commons.rede2_positive).count()[0]
    if rede < 3:
        sampledNaoEvento = sampledIndex.groupby("rede2").get_group(commons.rede2_negative).count()[0]
    sampledTotal     = sampledDuto + sampledNaoDuto
    naoDutoPercent   = sampledNaoDuto/sampledTotal*100
    dutoPercent      = sampledDuto/sampledTotal*100
    eventoPercent    = sampledEvento/sampledTotal*100
    naoEventoPercent = sampledNaoEvento/sampledTotal*100

    if rede == 1:
        negLabelName = commons.rede1_negative
        posLabelName = commons.rede1_positive
        cumNeg     = manualIndex.groupby("rede1").get_group('Nada').count()[0]+\
                     manualIndex.groupby("rede1").get_group('Confuso').count()[0]
        cumPos     = manualIndex.groupby("rede1").get_group(commons.rede1_positive).count()[0]
        
        # Exception for case where there are no positive or negative images automatically annotated
        if commons.rede1_negative in set(autoIndex['rede1'].values):
            autoNeg    = autoIndex.groupby("rede1").get_group(commons.rede1_negative).count()['rede1']
        else:
            autoNeg = 0
        if commons.rede1_positive in set(autoIndex['rede1'].values):
            autoPos    = autoIndex.groupby("rede1").get_group(commons.rede1_positive).count()['rede1']
        else:
            autoPos = 0
    elif rede == 2:
        negLabelName = commons.rede2_negative
        posLabelName = commons.rede2_positive
        cumNeg     = manualIndex.groupby("rede2").get_group(commons.rede2_negative).count()[0]
        cumPos     = manualIndex.groupby("rede2").get_group(commons.rede2_positive).count()[0]

        # Exception for case where there are no positive or negative images automatically annotated
        if commons.rede2_negative in set(autoIndex['rede2'].values):
            autoNeg    = autoIndex.groupby("rede2").get_group(commons.rede2_negative).count()['rede2']
        else:
            autoNeg = 0
        if commons.rede2_positive in set(autoIndex['rede2'].values):
            autoPos    = autoIndex.groupby("rede2").get_group(commons.rede2_positive).count()['rede2']
        else:
            autoPos = 0
    elif rede == 3:
        negLabelName = "Nao"+target_class
        posLabelName = target_class
        
        sampledClassPos = sampledIndex.groupby("rede3").get_group(posLabelName).count()[0]
        sampledClassNeg = sampledIndex.groupby("rede2").get_group(commons.rede2_positive).count()[0] - sampledClassPos
        
        sampledTotal    = sampledIndex.shape[0]
        sampleNegPercent    = sampledClassNeg/sampledTotal*100
        samplePosPercent    = sampledClassPos/sampledTotal*100

        cumPos     = manualIndex.groupby("rede3").get_group(posLabelName).count()[0]
        cumNeg     = manualIndex.groupby("rede2").get_group(commons.rede2_positive).count()[0] - cumPos

        # Exception for case where there are no positive or negative images automatically annotated
        if posLabelName in set(autoIndex['rede3'].values):
            autoPos    = autoIndex.groupby("rede3").get_group(posLabelName).count()['rede3']
        else:
            autoPos = 0
        autoNeg    = autoIndex.groupby("rede2").get_group(commons.rede2_positive).count()[0] - autoPos

    cumTotal         = cumPos + cumNeg
    cumNegPercent    = cumNeg/cumTotal*100
    cumPosPercent    = cumPos/cumTotal*100

    autoLabel        = autoIndex.shape[0]
    autoLabelPercent = autoLabel/numUnlabel*100

    # Compose manual image distribution string
    distributionString = "Manual annotation distribution:\n"
    if (rede == 1) or (rede == 2):
        distributionString +=\
        "NaoDuto:      {} images ({:.2f} %)\n\
        Duto:         {} images ({:.2f} %)\n\
            NaoEvento {} images ({:.2f} %)\n\
            Evento:   {} images ({:.2f} %)\n\
        Total:        {} images (100%)\n".format(sampledNaoDuto, naoDutoPercent, sampledDuto, dutoPercent,
                                                sampledNaoEvento, naoEventoPercent, sampledEvento, eventoPercent,
                                                sampledTotal)
    if rede == 3:
        distributionString +=\
            "{}:\t{} images ({:.2f} %)\n\
            {}:\t\t{} images ({:.2f} %)\n\
            Total\t\t{} images (100 %)\n".format(posLabelName, sampledClassPos, samplePosPercent,
                                               negLabelName, sampledClassNeg, sampleNegPercent,
                                               sampledTotal)

    # Assemble report string
    reportString = "Rede{}.\n{} unlabeled images remain. Sampled {} images for manual annotation.\n".format(rede,
                                                                                 numUnlabel, numSampled)+\
    distributionString+\
"Cumulative manual annotation distribution:\n\
    {}:      {} images ({:.2f} %)\n\
    {}:      {} images ({:.2f} %)\n\
    Total:   {} images (100%)\n".format(negLabelName, cumNeg, cumNegPercent,
                                        posLabelName, cumPos, cumPosPercent, cumTotal)+\
"Train Hyperparams:\n\
    Num Epochs:        {}\n\
    Batch Size:        {}\n\
    Optimizer:         Adam\n\
Train Results:\n\
    Elapsed Time:      {}m\n\
    Best val loss:     {:.4f}\n\
    Best val accuracy: {:.2f} %\n".format(1,2,3,4,5)+\
"Thresholds val (99% pos ratio):\n\
    Upper 99% positive ratio: {:.4f}, {:.2f} % ground truth positives\n\
    Lower  1% positive ratio: {:.4f}, {:.2f} % ground truth positives\n\
    Validation:  {}/{} = {:.2f} % images annotated\n\
Automatic Annotation:\n\
    Imgs Positivas: {}; Imgs Negativas: {}\n\
    {}/{} = {:.2f} % imagens anotadas automaticamente\n".format(1.,2.,3.,4.,5.,6.,7., autoPos, autoNeg,
                                                            autoLabel,numUnlabel, autoLabelPercent)
                                                            # TODO: Add train info
    # Write report
    # with open(report_path, 'w') as f:
    #     f.write(reportString)
    utils.write_string(reportString, report_path, mode='w')
    if show:
        print(reportString)
    return reportString


# Automatic labeling
def automatic_labeling(outputs, outputs_index, unlabeled_index, upper_thresh, lower_thresh, rede,
                        target_class=None, verbose=True):
    '''
        Return a DataFrame whose entries are taken from unlabeled_index according to calculated indexes.
        The indexes are chosen so that their outputs are either above the upper threshold or below the lower.
    '''
    upperIndexes, lowerIndexes = get_auto_label_indexes(outputs, outputs_index, upper_thresh,
                                                        lower_thresh, verbose=True)
    
    autoIndex = get_classified_index(unlabeled_index, upperIndexes, lowerIndexes, rede,
                                    index_col="FrameHash", target_class=target_class, verbose=False)
    return autoIndex


def get_auto_label_indexes(outputs, outputs_index, upper_thresh, lower_thresh, verbose=True):
    datasetLen      = len(outputs)
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


def get_classified_index(index, pos_hashes, neg_hashes, rede, target_class=None, index_col="FrameHash",
                         verbose=True):
    '''
        Create new auto labeled index from the unlabeled_images index and positive and negative indexes
        lists.
    '''
    if index_col is not None:
        index.set_index("FrameHash", drop=False, inplace=True)

    if rede >= 1:
        positiveLabel1 = commons.rede1_positive
        negativeLabel1 = commons.rede1_negative
    if rede >= 2:
        positiveLabel2 = commons.rede2_positive
        negativeLabel2 = commons.rede2_negative
    if rede >= 3:
        assert target_class in commons.rede3_classes.values(), "Unknown target_class value."
        positiveLabel3 = target_class

    newPositives = index.reindex(labels=pos_hashes, axis=0, copy=True)
    newNegatives = index.reindex(labels=neg_hashes, axis=0, copy=True)

    datasetLen   = len(index)
    lenPositives = len(newPositives)
    lenNegatives = len(newNegatives)

    # Set positive and negative class labels
    if rede == 1:
        newPositives["rede1"] = [positiveLabel1]*lenPositives
        newNegatives["rede1"] = [negativeLabel1]*lenNegatives
    if rede == 2:
        newPositives["rede1"] = [positiveLabel1]*lenPositives
        newNegatives["rede1"] = [positiveLabel1]*lenNegatives

        newPositives["rede2"] = [positiveLabel2]*lenPositives
        newNegatives["rede2"] = [negativeLabel2]*lenNegatives
    if rede == 3:
        newPositives["rede1"] = [positiveLabel1]*lenPositives
        newNegatives["rede1"] = [positiveLabel1]*lenNegatives

        newPositives["rede2"] = [positiveLabel2]*lenPositives
        newNegatives["rede2"] = [positiveLabel2]*lenNegatives

        newPositives["rede3"] = [positiveLabel3]*lenPositives

    newLabeledIndex = pd.concat([newPositives, newNegatives], axis=0, sort=False)

    if rede == 2:
        newPositives["rede1"] = [positiveLabel1]*lenPositives

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
        lowerThreshList = copy(upperThreshList)[::-1]
    else:
        lowerThreshList = np.arange(0., 1., resolution)
        upperThreshList = np.arange(1., 0., -resolution)
    #     upperThreshList = np.arange(0., 1., resolution)
    #     lowerThreshList = np.arange(1., 0., -resolution)

    # Find upper threshold
    idealUpperThresh = find_ideal_upper_thresh(
                                    val_outputs, labels, upperThreshList, ratio=upper_ratio)#, verbose=True)

    # Find lower threshold
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
        get_auto_label_indexes(val_outputs, val_indexes, idealUpperThresh, idealLowerThresh, verbose=True)

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


## Dataset files manipulation
def df_to_csv(dataframe, save_path, verbose=True):
    dirs.create_folder(Path(save_path).parent)
    dataframe.to_csv(save_path, index=False)

    if verbose:
        print("Saved DataFrame to ", save_path)


def get_ref_dataset_val_video_list(folder_path, verbose=False):
    '''
        Get a list of video hashes from a dataset folder with a specific file tree.
        folder_path/
            xxx/hash1/
            yyy/hash2/
            ...
        
        Returns non-duplicated list of found hashes.
    '''
    globString = str(folder_path)+"/**"
    folderList = glob(globString, recursive=True)
    videoList = []
    for pathEntry in folderList:
        relString = Path(pathEntry).relative_to(folder_path)
        if len(relString.parts) == 2:
            videoHash = relString.parts[-1]
            videoList.append(videoHash)
    videoList = list(set(videoList))

    return videoList


def split_validation_set_from_video_list(df_path, index_list, key_column="HashMD5", verbose=False):
    '''
        Split a DataFrame given by df_path in two, according to index_list. The DataFrame is split in
        two other: one containing only entries with indexes in index_list; the other is the converse,
        containing none of the given indexes.

        Arguments:
        df_path: str filepath
            Filepath to target DataFrame saved in csv format.

        index_list: list
            List of indices to guide the split. One split set will contain only entries with indexes
            in this list and the other set will contain the remaining entries.

        key_column: str
            Name of the DataFrame column where the indexes of index_list will be searched.

        Returns:
        trainIndex: DataFrame
            DataFrame subset from input DataFrame. Contains only entries with indexes not present in 
            index_list.

        valIndex: DataFrame
            DataFrame subset from input DataFrame. Contains only entries with indexes present in 
            index_list.
    '''
    index = pd.read_csv(df_path)
    index.dropna(axis=0, subset=[key_column], inplace=True)

    valHash   = index_list
    trainHash = set(index[key_column]) - set(valHash)
    # valHash = utils.compute_file_hash_list(index_list)

    index.set_index(key_column, drop=False, inplace=True)
    trainIndex = index.loc[trainHash, :].copy()
    valIndex = index.loc[valHash, :].copy()
    trainIndex.reset_index(inplace=True, drop=True)
    valIndex.reset_index(inplace=True, drop=True)
    
    return trainIndex, valIndex


def merge_indexes(index_path_list, key_column):
    '''
        Read a list of DataFrame paths, concatenates them and remove duplicated elements from resulting DF.
    '''
    # assert (len(index_path_list) >= 2) and \
    #        not(isinstance(index_path_list, str)), \
    # "Argument index_path_list must be a list of two or more DataFrame paths."
    assert hasattr(index_path_list, "__iter__") and \
           not(isinstance(index_path_list, str)), \
            "Argument index_path_list must be a list of two or more DataFrame paths."

    indexListNoDups = [remove_duplicates(pd.read_csv(x), key_column) for x in index_path_list]
    
    if len(indexListNoDups) > 1:
        newIndex = pd.concat(indexListNoDups, axis=0, sort=False)
    else:
        newIndex = indexListNoDups[0]
    newIndex = remove_duplicates(newIndex, key_column)
    return newIndex

def start_loop(prev_annotated_path, target_class, target_column, verbose=True):
    '''
        Splits previous annotated image index in auto and manual labeled indexes.
        Creates first iteration folder.
    '''
    iter1Folder = Path("/".join(prev_annotated_path.parts[:-2])) / "iteration_1"
    newUnlabeledPath = Path(prev_annotated_path).with_name("unlabeled_images_iteration_0.csv")
    newReferencePath = Path(prev_annotated_path).with_name("reference_images.csv")
    newLabeledPath   = iter1Folder / "sampled_images_iteration_1.csv"
    dirs.create_folder(iter1Folder)

    prevAnnotated = pd.read_csv(prev_annotated_path)

    # Create nextLevelIndex df with only images that have been annotated as target_class in the
    # previous iteration. Save as reference index for this loop
    mask = prevAnnotated[target_column] == target_class
    nextLevelIndex = prevAnnotated.loc[mask, :]
    nextLevelIndex = remove_duplicates(nextLevelIndex, "FrameHash", verbose=True)
    
    nextLevelIndex.to_csv(newReferencePath, index=False)
    
    # New unlabeled set unlabeled_images_iteration_0 is actually composed of all images
    # newUnlabeled  = nextLevelIndex.copy()
    newUnlabeled  = nextLevelIndex.groupby("Annotation").get_group('auto') # To get only auto annotated images
    
    # Save manual labeled images as sampled_images for first iteration
    newLabeled = nextLevelIndex.groupby("Annotation").get_group('manual')
    
    newUnlabeled.to_csv(newUnlabeledPath, index=False)
    newLabeled.to_csv(newLabeledPath, index=False)

    if verbose:
        print("Annotated last level: ", prevAnnotated.shape)
        print("To be used in current step: ", nextLevelIndex.shape)
        print("Unlabeled: ", newUnlabeled.shape)
        print("Labeled:   ", newLabeled.shape)
    return newUnlabeled, newLabeled


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

        The operation performed can be interpreted as a set complement between reference and
        to_drop DataFrames. Returns a DataFrame with length equal to (len(reference_df) - len(to_drop_df)).
    '''
    # Drop NaNs from allAnnotations; TODO: Find out how NaNs could appear in FrameHash column
    print("Number of NaNs removed in final_annotated_images: ", to_drop_df[column_label].isna().sum())
    if to_drop_df[column_label].isna().sum() > 10:
        print("\nWarning! High number of NaNs! Check if everything is normal.\n")
    to_drop_df.dropna(subset=[column_label], inplace=True)

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


def copy_dataset_to_folder(index_path, dest_folder, path_column="FramePath", verbose=True):
    ''' 
        Move files to dest_folder. File paths are given in the path_column of a DataFrame
        saved to index_path.

        Files are read from source path and copied to dest_folder keeping the original filenames.

        index_path: str filepath
            Filepath of a DataFrame saved in csv format. DataFrame path_column field must contain 
            the valid filepaths.
        
        dest_folder: str folder path
            Path to destination folder.
        
        path_column: str
            Name of DataFrame field containing the source filepaths.
    '''
    def _add_folder_and_copy(x):
        return utils.copy_files(Path(x), dest_folder / Path(x).name)
    dirs.create_folder(dest_folder)
    index = pd.read_csv(index_path, low_memory=False)

    if verbose:
        print("\nMoving {} files from dataset folder to sampled images folder...".format(len(index)))
    
    successes = np.sum(index[path_column].map(_add_folder_and_copy))
    if verbose:
        print("{}/{} files copied.".format(successes, len(index[path_column])))
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


def move_to_class_folders(index_path, image_folder_path, target_net="rede1", target_class=None, move=True,
                             skip_untranslated=True, verbose=False):
    '''
        Sort labeled images in class folders according to index file with labels and filepaths.

        Arguments:
        index_path: str filepath

        image_folder_path: str folder path

        target_net: str

        target_class: str or None

        move: bool
            If True, move images from source to new folder. If False, just copy and don't move them.

        skip_untranslated: bool
            If True, ignore images translated to commons.no_translation.
    '''
    index_path     = Path(index_path)
    dirs.create_folder(image_folder_path)
    # assert image_folder_path.is_dir(), "Folder argument must be a valid image folder."
    
    # If target net is rede3, check if target_class is valid, if provided
    if (target_net == commons.net_target_column[3]) and (target_class is not None):
        assert target_class in commons.rede3_classes.values(), "Invalid target class for rede3."

    imageIndex = pd.read_csv(index_path)
    numImages  = len(imageIndex)

    # Get unique tags and create the respective folders
    tags = set(imageIndex[target_net]) - set("-")

    print("Found tags ", tags)
    for tag in tags:
        tag = translate_labels(tag, target_net, target_class=target_class)
        
        if skip_untranslated and tag == commons.no_translation:
            continue
        
        dirs.create_folder(image_folder_path / tag)
        if verbose:
            print("Created folder ", (image_folder_path / tag))

    if move:
        print("Moving files to class folders...")
    else:
        print("Copying files to class folders...")

    destList = []
    indexesToDrop = []
    for i in tqdm(range(numImages)):
        tag      = translate_labels(imageIndex.loc[i, target_net], target_net, target_class=target_class)
        
        if skip_untranslated and tag == commons.no_translation:
            indexesToDrop.append(imageIndex.index[i])
            continue
        
        # Get source path
        imageName  = str(imageIndex.loc[i, 'FrameName'])
        source   = image_folder_path / imageName
        # Get dest path
        destName = Path(tag) / imageName
        dest     = image_folder_path / destName

        if verbose:
            print("Moving\n{}\nto\n{}".format(source, dest))

        if move:
            sh.move(source, dest)
        else:
            utils.copy_files(source, dest)
        destList.append(dest)
    
    # Drop ignored entries
    imageIndex.drop(labels=indexesToDrop, axis=0, inplace=True)
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
        if Path(f).is_dir():
            sh.rmtree(f)
        elif Path(f).is_file():
            os.remove(f)

    if index is not None: # Update frame paths in index
        print("\nSaving to index...")
        def get_parts(x): return "/".join(x.parts[-3:])

        trainHashList   = utils.make_videos_hash_list(trainDestList, hash_column="FrameHash")["FrameHash"]
        valHashList     = utils.make_videos_hash_list(valDestList, hash_column="FrameHash")["FrameHash"]

        trainDestList   = list(map(get_parts, trainDestList))
        valDestList     = list(map(get_parts, valDestList))

        index.set_index('FrameHash', drop=False, inplace=True)
        # TODO: Check if train dest paths are guaranteed to be saved in
        #  the same order as index. If not, find other way
        ## Note: Iterating the index via for statement is unfeasibly slow
        trainIndex              = index.reindex(labels=trainHashList, axis=0, copy=True)
        trainIndex['TrainPath'] = trainDestList
        trainIndex['set']       = ['train']*setLengths[0]

        valIndex                = index.reindex(labels=valHashList, axis=0, copy=True)
        valIndex['TrainPath']   = valDestList
        valIndex['set']         = ['val']*setLengths[1]

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

        Adds a column defined in commons.FRAME_HASH_COL_NAME with file hashes of the images found in 
        filepaths at framePathColum column.
    '''
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
    labelsDf[commons.FRAME_HASH_COL_NAME] = utils.make_videos_hash_list(labelsDf['FramePath'])['HashMD5']

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


class Rede3Translator:
    '''
        Translate normalized class labels to binary classification labels.
    '''
    def __init__(self, target_class):
        self.target_class = target_class

    def translate_label(self, label_name):
        if label_name.lower() == self.target_class.lower():
            translatedLabel = self.target_class
        else:
            translatedLabel = "Nao"+self.target_class
        return translatedLabel


def translate_labels(labels, target_net, target_class=None, verbose=False):
    '''
        Translate interface-generated labels to the index standard, following commons.classes
        class list. Performs label normalization on input labels. If target_net is rede3, can
        also perform translation to binary labels.

        Argument:
            label: string or list of strings
                A string or list of strings representing one or more classes, following
                the pattern of interface-generated labels.
            
            target_net: str
                Must one of the set ('rede1', 'rede2', 'rede3').

            target_class: str
                Valid only for target_net == 'rede3'. Must be a class defined in commons.rede3_classes.
                If None is passed, the function will skip translation to binary labels and will only
                perform label normalization.

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
            # Translate class labels as task-relevant binary labels
            innerTranslation = commons.net_binary_table[target_net][translatedLabel]
        else:
            if verbose:
                warnings.warn("\nTranslation not found for label:\n\t{}".format(label))
            innerTranslation = commons.no_translation

        # If target net is rede3, translate normalized labels to binary labels
        if (target_net == commons.net_target_column[3]) and (target_class is not None):
            rede3Translator = Rede3Translator(target_class)
            return rede3Translator.translate_label(innerTranslation)
        else:
            return innerTranslation

    # This table formats and normalizes manually annotated class labels
    # Fixes a limited number of common spelling mistakes
    translationTable = commons.net_class_translation_table

    if isinstance(labels, str):
        # If not list, just translate input
        translation = _translate(labels)
    elif hasattr(labels, "__iter__"):
        # If list, apply translation subroutine to every element in list
        translation = list(map(_translate, labels))
    else:
        raise ValueError("Input must be a string or list of strings.")
    
    return translation
