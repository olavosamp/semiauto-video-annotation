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

''' Get class counts from rede 3 dataset csv file resulting from fuse_binary_datasets script '''
# rede = int(input("\nEnter desired net number.\n"))
rede = 3
classList = commons.rede3_classes

compiledPositivesPath = Path(dirs.iter_folder) / "dataset_rede_{}_positives_binary.csv".format(rede)

datasetDf = pd.read_csv(compiledPositivesPath)

datasetGroup = datasetDf.groupby('rede3')

print(datasetGroup.count()['FrameHash'])
countDf = pd.DataFrame(datasetGroup.count()['FrameHash'])
countDf['Counts'] = countDf['FrameHash']
total = countDf['Counts'].sum()
countDf['Percentage'] = countDf['Counts'].apply(lambda x: x/total)

print(countDf)
print(total)
countDf.drop("FrameHash", axis=1, inplace=True)
countDf.to_excel(compiledPositivesPath.with_name("semiauto_class_counts.xlsx"))