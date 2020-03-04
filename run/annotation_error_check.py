import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm

import libs.dirs as dirs
import libs.utils as utils
import libs.dataset_utils as dutils
import models.utils as mutils
import libs.commons as commons
from libs.vis_functions import plot_confusion_matrix

''' Compares automatic annotations and error check annotations to obtain
a labeling error measure'''

def check_labeling_error(label1, label2, target_label):
    if (label1 == target_label) and (label1 == label2):
        errorCheck = False
    elif (label1 != target_label) and (label2 != target_label):
        errorCheck = False
    else:
        errorCheck = True
    return errorCheck

annotationFolder = (Path(dirs.images) / "samples_error_check_felipe")
foldersRede3 = glob(str(annotationFolder)+"/*rede_3*/")
foldersResto = [str(annotationFolder / "full_dataset_rede_1"),
                str(annotationFolder / "full_dataset_rede_2")
]

netFolders = []
netClass   = []

netFolders.extend(foldersResto)
netClass.extend([commons.rede1_positive.lower(),
                 commons.rede2_positive.lower()])

for elem in foldersRede3:
    netFolders.append(elem)
    netClass.append(str(Path(elem).name).split("_")[-1])

for classFolder, targetClass in zip(netFolders, netClass):
    rede = classFolder[str(classFolder).find('rede_')+5]
    print("rede "+str(rede))

    classFolder = Path(classFolder)
    # targetClass = str(classFolder.name).split("_")[-1]

    print("\nProcessing class ", classFolder.name)
    errorDf = pd.read_csv(classFolder / "sampled_images_corrections.csv")
    autoDf  = pd.read_csv(classFolder / "automatic_check_index.csv")

    print("Calculating hashes...")
    errorDf["FrameHash"] = utils.compute_file_hash_list(errorDf.loc[:, "imagem"],
                                folder=(classFolder / "automatic") / "sampled_images")

    dutils.df_to_csv(errorDf, classFolder / "sampled_images_corrections_results.csv", verbose=False)
    autoDf.set_index("FrameHash", inplace=True, drop=False)
    errorDf.set_index("FrameHash", inplace=True, drop=False)
    errorDf['error_check'] = [None]*len(errorDf)
    
    for i in autoDf.index:
        errorTag = str(errorDf.loc[i, 'rede'+rede]).lower()
        autoTag  = str(autoDf.loc[i, 'rede'+rede]).lower()

        if (autoTag == targetClass) and (autoTag == errorTag):
            errorCheck = False
        elif (autoTag != targetClass) and (errorTag != targetClass):
            errorCheck = False
        else:
            errorCheck = True
        
        errorCheck = check_labeling_error(autoTag, errorTag, targetClass)
        
        errorDf.loc[i, 'error_check'] = errorCheck
    errorPercent = errorDf['error_check'].sum()/len(errorDf)*100
    print("\nMatched labels: {}/{} labels.".format(len(errorDf) - errorDf['error_check'].sum(), len(errorDf)))
    print("Error percentage: {:.2f}%".format(errorPercent))
    # input()
