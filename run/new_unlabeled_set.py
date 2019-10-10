import numpy                as np
import pandas               as pd
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from libs.index             import IndexManager

def subset_index(main_index, subset_index):
    '''
        Select entries from main_index DataFrame corresponding to the indexes of
        subset_index DataFrame. Index column is FrameHash.
    '''
    main_index.set_index('FrameHash', drop=False, inplace=True)
    resultIndex = main_index.loc[subset_index["FrameHash"], :].copy()

    print(resultIndex.shape)
    return resultIndex


def index_complement(reference_df, to_drop_df, column_label):
    '''
        Drop rows from 'reference_df' DataFrame indicated by column_label
        column of 'to_drop_df' DataFrame.
    '''
    reference_df.set_index(column_label, drop=False, inplace=True)
    reference_df.drop(labels=to_drop_df[column_label], axis=0, inplace=True)

    # print(reference_df.shape)

    reference_df.reset_index(drop=True, inplace=True)
    return reference_df.copy()


# unlabelIndexPath    = Path(dirs.index) / "unlabeled_index_2019-8-18_19-32-37_HASHES.csv"
# sampledIndexPath    = Path(dirs.iter_folder)/ "full_dataset/iteration_0/sampled_images_iteration_0.csv"

unlabelIndexPath    = Path(dirs.index) / "unlabeled_index_2019-8-18_19-32-37_HASHES.csv"
sampledIndexPath    = Path(dirs.iter_folder)/ "full_dataset/iteration_0/final_annotated_images_iteration_1.csv"

newUnlabelIndexPath = Path(dirs.iter_folder)/ "full_dataset/iteration_2/unlabeled_images_iteration_2.csv"

# Load model outputs and unlabeled images index
# indexUnlabel = IndexManager(unlabelIndexPath)
# indexSampled = IndexManager(sampledIndexPath)
indexUnlabel = pd.read_csv(unlabelIndexPath)
indexSampled = pd.read_csv(sampledIndexPath)
print(indexUnlabel.index.shape)

newIndex = index_complement(indexUnlabel, indexSampled, "FrameHash")
print(newIndex.shape)

# indexUnlabel.write_index(newUnlabelIndexPath, prompt=False, backup=False)
dirs.create_folder(newUnlabelIndexPath.parent)
newIndex.to_csv(newUnlabelIndexPath, index=False)

