import numpy                as np
import pandas               as pd
from glob                   import glob
from pathlib                import Path

import libs.dirs            as dirs
# import libs.utils           as utils
# import models.utils         as mutils
import libs.dataset_utils   as dutils
import libs.commons         as commons

tempLabels = list(commons.rede3_classes.values())
tempLabels.extend(['Duto', 'Nao_Duto', 'Evento', 'Nao_Evento', 'duct', 'not_duct'])
classLabels = [f.lower() for f in tempLabels]

def get_class(filepath):
    filepath = Path(filepath).relative_to(dirs.dataset)
    return str(filepath.parts[2]).lower()

def get_set(filepath):
    filepath = Path(filepath).relative_to(dirs.dataset)
    return str(filepath.parts[1].lower())

def check_valid_class(filepath):
    fileClass = get_class(filepath)
    if fileClass in classLabels:
        return True
    else:
        return False


if __name__ == "__main__":
    # dfList = []
    allDatasets = None
    for net_type in commons.network_types.values():
        for val_type in commons.val_types.values():
            for rede in commons.net_target_column.keys():
                # Dataset root folder
                datasetPath = Path(dirs.dataset) / \
                    "{}_dataset_rede_{}_val_{}".format(net_type, rede, val_type)
                print("\n", datasetPath.stem)
                # Get file list
                globString = str(datasetPath)+"/**/*jpg"
                fileList = glob(globString, recursive=True)
                print(globString)

                entryDf = pd.DataFrame({'FramePath': fileList})

                # Drop unkown classes
                toKeep = entryDf['FramePath'].apply(check_valid_class)
                entryDf = entryDf.loc[toKeep, :]
                fileLen = entryDf.shape[0]

                entryDf['Class'] = entryDf['FramePath'].apply(get_class)
                entryDf['Rede'] = [rede]*fileLen
                entryDf['Validation'] = [val_type]*fileLen
                entryDf['Dataset'] = [net_type]*fileLen
                entryDf['Set'] = entryDf['FramePath'].apply(get_set)

                print(entryDf.groupby('Class').count())
                # input()
                if allDatasets is None:
                    allDatasets = entryDf
                else:
                    allDatasets = pd.concat([allDatasets, entryDf], ignore_index=True)
                # dfList.append(entryDf)
    # print(dfList)
    # print(len(dfList))
    # allDatasets = pd.concat([dfList], ignore_index=True)
    print(allDatasets.groupby('Rede').count())
    print()
    print(allDatasets.groupby('Dataset').count())
