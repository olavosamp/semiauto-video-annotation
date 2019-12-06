import numpy                as np
import pandas               as pd
# from tqdm                   import tqdm
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

# Get final annotation indexes of all classes
indexDict = {}
for classLabel in classList.values():
    indexPath  = str(Path(dirs.iter_folder) / "full_dataset_rede_3_{}".format(classLabel.lower()) /\
        "final_annotated_images_full_dataset_rede_3_{}.csv".format(classLabel.lower()))
    indexDict[classLabel] = pd.read_csv(indexPath, low_memory=False)

for key in indexDict.keys():
    indexLen = len(indexDict[key])
    print("Positives before: ", np.sum(indexDict[key]['rede3'] == key))
    print("Negatives before: ", np.sum(indexDict[key]['rede3'] != key))
    labelList = indexDict[key]['rede3'].copy().map(str)
    indexDict[key]['rede3'] = dutils.translate_labels(labelList,
                                                      'rede3',
                                                      target_class=key)
    
    print("Positives after: ", np.sum(indexDict[key]['rede3'] == key))
    print("Negatives after: ", np.sum(indexDict[key]['rede3'] != key))
    input()


# TODO:
# for each index, translate rede3 labels to binary labels;
# split positive and negative entries;
# each positive set will correspond to a class;
# images that are labeled negative for all classes are split in a negatives index;
# fuse the 5 indexes in a single index, using the priority table;
# concatenate negatives index and new fusion index;
# check if before and after sizes matches

compiledIndex = pd.concat(indexDict)
print(compiledIndex.shape)

duplicatesMask = compiledIndex.duplicated(subset=commons.FRAME_HASH_COL_NAME, keep=False)

# Split compiled index in duplicated and non-duplicated entries
duplicatesIndex = compiledIndex.loc[duplicatesMask, :]
compiledIndex   = compiledIndex.loc[~duplicatesMask, :]

def _translate_dup_label(label_list):
    for priority_label in commons.rede_3_multiclass_priority_table:
        if priority_label in label_list:
            return priority_label
    return None

groups = duplicatesIndex.groupby(by=commons.FRAME_HASH_COL_NAME)
print(groups.groups)
# for group in groups.groups.keys():
