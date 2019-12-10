import numpy                as np
import pandas               as pd
from tqdm                   import tqdm
from glob                   import glob
from pathlib                import Path
# from copy                   import copy

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
import libs.commons         as commons

''' Fuse binary class datasets in a multiclass dataset '''
rede = 3
classList = commons.rede3_classes

compiledPositivesPath = Path(dirs.iter_folder) / "dataset_rede_3_positives_binary.csv"

# Get final annotation indexes of all classes
# Arrange them in a dictionary with class names as keys and index DataFrames as values
indexDict = {}
for classLabel in classList.values():
    indexPath  = str(Path(dirs.iter_folder) / "full_dataset_rede_3_{}".format(classLabel.lower()) /\
        "final_annotated_images_full_dataset_rede_3_{}.csv".format(classLabel.lower()))
    indexDict[classLabel] = pd.read_csv(indexPath, low_memory=False)

# Get only rede2 positive entries (Eventos)
# PositivesList dict values are the DataFrames indexes of each class;
# each DF contains only rede2 positive entries and have binary-translated rede3 labels
positivesList = {}
for key in indexDict.keys():
    indexLen = len(indexDict[key])
    labelList = indexDict[key]['rede3'].copy().map(str) # Get a copy of rede3 elements and typecast to string
    # Translate rede3 labels to binary labels
    indexDict[key]['rede3'] = dutils.translate_labels(labelList,
                                                      'rede3',
                                                      target_class=key)
    mask = indexDict[key]['rede3'] == key
    positivesList[key] = indexDict[key].loc[mask, :]


# TODO:
# V for each index, translate rede3 labels to binary labels;
# V split positive and negative entries;
# V each positive set will correspond to a class;
# V fuse the 5 indexes in a single index
# V translate duplicated labels using the priority table;
# images that are labeled negative for all classes are split in a negatives index;
# concatenate negatives index and new fusion index;
# check if before and after sizes matches

print("\nConcatenating positive indexes...")
compiledPositivesIndex = pd.concat(positivesList.values())

# allIndexes = indexDict.values()[0].index # Assumes all classes contain all images
# compiledPositivesIndex = dutils.remove_duplicates(compiledPositivesIndex, commons.FRAME_HASH_COL_NAME)
# negMask = set(compiledPositivesIndex.index) - set(allIndexes)
# negativesIndex = 


def _translate_dup_label(label_list):
    for priority_label in commons.rede_3_multiclass_priority_table:
        if priority_label in label_list:
            return priority_label
    return None

# Split compiled index in duplicated and non-duplicated entries
duplicatesMask  = compiledPositivesIndex.duplicated(subset=commons.FRAME_HASH_COL_NAME, keep=False)
duplicatesIndex = compiledPositivesIndex.loc[duplicatesMask, :]
frameGroup = duplicatesIndex.groupby(commons.FRAME_HASH_COL_NAME)

print("\nTranslating all duplicates...")
compiledPositivesIndex = dutils.remove_duplicates(compiledPositivesIndex, commons.FRAME_HASH_COL_NAME)
compiledPositivesIndex.set_index('FrameHash', drop=False, inplace=True)
for frameHash, group in tqdm(frameGroup):
    newLabel = _translate_dup_label(group['rede3'].values)
    compiledPositivesIndex.loc[frameHash, 'rede3'] = newLabel

compiledPositivesIndex.reset_index(drop=True, inplace=True)

print(compiledPositivesIndex['rede3'][:30])

compiledPositivesIndex.to_csv(compiledPositivesPath, index=False)

# groups = duplicatesIndex.groupby(by=commons.FRAME_HASH_COL_NAME)
# print(groups.groups)
# for group in groups.groups.keys():
