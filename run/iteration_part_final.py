import numpy                as np
import pandas               as pd
from glob                   import glob
from copy                   import copy
from tqdm                   import tqdm
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
import libs.commons         as commons
# from libs.index             import IndexManager

# datasetName = "full_dataset_rede_3_anodo"

rede      = int(input("Enter net number.\n"))

if rede == 3:
    target_class = dutils.get_input_target_class(commons.rede3_classes)
    datasetName  = "full_dataset_rede_{}_{}".format(rede, target_class.lower())
else:
    target_class = None
    datasetName  = "full_dataset_rede_{}".format(rede)

sampledImagesPath       = Path(dirs.images) / "{}_results_samples".format(datasetName)
loopFolder              = Path(dirs.iter_folder) / datasetName
referenceIndexPath      = loopFolder / "iteration_0/reference_images.csv"
previousLevelIndexPath  = loopFolder / "iteration_0/unlabeled_images_iteration_0.csv"
autoIndexFullPath       = loopFolder / "final_automatic_images_{}.csv".format(datasetName)
manualIndexFullPath     = loopFolder / "final_manual_images_{}.csv".format(datasetName)
annotatedIndexFullPath  = loopFolder / "final_annotated_images_{}.csv".format(datasetName)
reportPath              = loopFolder / "annotation_report.txt"
binaryDatasetPath       = Path(dirs.iter_folder) / "dataset_rede_{}_eval_setup.csv".format(rede)

referenceIndex     = pd.read_csv(referenceIndexPath, low_memory=False)
referenceIndex     = dutils.remove_duplicates(referenceIndex, "FrameHash")

previousLevelIndex = pd.read_csv(previousLevelIndexPath, low_memory=False)
previousLevelIndex = dutils.remove_duplicates(previousLevelIndex, "FrameHash")


# Get list of all iteration folders
folderList = utils.make_path( glob(str(loopFolder)+"/iteration*"))
tempList = copy(folderList)
for path in tempList:
    if not(path.is_dir()):
        folderList.remove(path)

if rede == 3:
    print("\nNet {}, target class {}.".format(rede, target_class))
else:
    print("\nNet {}.".format(rede))
print("Ran {} iterations of annotation.".format(len(folderList)))

# Sort folder list by iteration
def _get_iter(path):
    # Function that gets integer iteration from string path
    return int(str(path).split("_")[-1])
folderList.sort(key=_get_iter)

# Group all automatic annotations
# Drop first and last iterations, as they don't have automatic annotations
iterList = list(range(len(folderList)))
autoIterList = iterList[1:-1]
autoIndexList = []
for i in tqdm(autoIterList):
    folder = folderList[i]
    autoIndexList.append(pd.read_csv(folder/ "automatic_labeled_images_iteration_{}.csv".format(i), low_memory=False))

# tot = 0
# for i in range(len(autoIndexList)):
#     # print(i+1)
#     index = autoIndexList[i]
#     print("iteration_{}: {} images".format(i+1, index.shape[0]))
#     tot += index.shape[0]
# print("Total: ", tot)
# exit()

# Concatenate and save auto annotated images
autoIndexFull = pd.concat(autoIndexList, axis=0, sort=False)

autoIndexFull = dutils.remove_duplicates(autoIndexFull, "FrameHash")
autoIndexFull.to_csv(autoIndexFullPath, index=False)

# Group all manual annotations
# Get cumulative manual index of second to last iteration (the last one with cumulative annotations)
cumManualIndex = pd.read_csv(folderList[-2] / \
                    "manual_annotated_images_iteration_{}_train_val_split.csv".format(iterList[-2]), low_memory=False)

# Process sampled image csv
# Fill index information of sampled images of the final iteration
lastFolder = folderList[-1]
sampledLastIterIndex = pd.read_csv(lastFolder / "sampled_images_iteration_{}.csv".format(iterList[-1]), low_memory=False)
sampledLastIterIndex["FrameHash"] = utils.compute_file_hash_list(sampledLastIterIndex["imagem"].values,
                                                        folder= lastFolder / "sampled_images")
manualLastIterIndex  = dutils.fill_index_information(referenceIndex, sampledLastIterIndex,
                                                        "FrameHash", [ 'rede1', 'rede2', 'rede3'])

# Concatenate and save manual annotated images
manualIndexFull = pd.concat([cumManualIndex, manualLastIterIndex], axis=0, sort=False)
manualIndexFull.to_csv(manualIndexFullPath, index=False)

# Add Annotation column to indexes
autoIndexFull["Annotation"]   = ['auto']*len(autoIndexFull)
manualIndexFull["Annotation"] = ['manual']*len(manualIndexFull)

# Concatenate and save all annotated images
annotatedIndexFull = pd.concat([manualIndexFull, autoIndexFull], axis=0, sort=False)
annotatedIndexFull = dutils.remove_duplicates(annotatedIndexFull, "FrameHash")
annotatedIndexFull.to_csv(annotatedIndexFullPath, index=False)

# Assemble, print and save report
# TODO: Encapsulate report function and overhaul dutils.make_report
if rede == 1:
    previousManualLen = 0
else:
    previousManualLen = pd.read_csv(folderList[1] / "sampled_images_iteration_1.csv", low_memory=False).shape[0]

autoLen        = autoIndexFull.shape[0]
manualLen      = manualIndexFull.shape[0] - previousManualLen # Manual anotations made only on this level
totalLen       = annotatedIndexFull.shape[0] - previousManualLen
totalManualLen = manualIndexFull.shape[0]

imgNumberStr = ("Ran {} iterations of annotation.".format(len(folderList)))
imgNumberStr += ("\n\nFinal number of annotated images in level {}:".format(rede))
imgNumberStr += ("\nAutomatic: {}\t({:.2f} %)".format(autoLen, autoLen/totalLen*100 ))
imgNumberStr += ("\nManual:    {}\t({:.2f} %)".format(manualLen, manualLen/totalLen*100 ))
imgNumberStr += ("\nTotal:     {}\t({:.2f} % of current level starting dataset)\n".format(totalLen,
                                                    totalLen/previousLevelIndex.shape[0]*100 ))

imgNumberStr += ("\n\nTotal number of manual annotated images (shared between levels):\n{} ({:.2f} % of\
 original dataset)".format(totalManualLen, totalManualLen/referenceIndex.shape[0]))

autoPos, autoNeg     = dutils.get_net_class_counts(autoIndexFullPath, rede, target_class=target_class)
manualPos, manualNeg = dutils.get_net_class_counts(manualIndexFullPath, rede, target_class=target_class)
totalPos, totalNeg   = dutils.get_net_class_counts(annotatedIndexFullPath, rede, target_class=target_class)

# Report strings
strAuto = ("\n\nClass distribution:")
strAuto += ("\nAutomatic:")
strAuto += ("\nPos:   {}\t({:.2f} %)".format(autoPos, autoPos/autoLen*100))
strAuto += ("\nNeg:   {}\t({:.2f} %)".format(autoNeg, autoNeg/autoLen*100))
strAuto += ("\nTotal: {}\t({:.2f} %)".format(autoNeg+autoPos, (autoNeg+autoPos)/autoLen*100))

strManual =("\nManual:")
strManual +=("\nPos:   {}\t({:.2f} %)".format(manualPos, manualPos/(manualLen+previousManualLen)*100))
strManual +=("\nNeg:   {}\t({:.2f} %)".format(manualNeg, manualNeg/(manualLen+previousManualLen)*100))
strManual +=("\nTotal: {}\t({:.2f} %)".format(manualNeg+manualPos, (manualNeg+manualPos)/(manualLen+previousManualLen)*100))

strTotal = ("\nTotal:")
strTotal += ("\nPos:   {}\t({:.2f} %)".format(totalPos, totalPos/(totalLen + previousManualLen)*100))
strTotal += ("\nNeg:   {}\t({:.2f} %)".format(totalNeg, totalNeg/(totalLen + previousManualLen)*100))
strTotal += ("\nTotal: {}\t({:.2f} %)".format(totalNeg+totalPos, (totalNeg+totalPos)/(totalLen + previousManualLen)*100))

print(strAuto)
print(strManual)
print(strTotal)
utils.write_string(strAuto+strManual+strTotal, reportPath, mode='w')

# Convert final_annotated_images to a net evaluation setup and copy to dirs.iter_folder
# Translate rede1 3 classes to 2 for binary classification
# Get only past level positive examples for rede2
# Get only positive class examples for rede3 (in other script)
netName = "rede"+str(rede)
if rede == 3:
    print("\nPlease run script \'run/fuse_binary_datasets.py\' to generate evaluation dataset.")
    exit()
elif rede == 2:
    # Get only positive examples from previous level
    mask = (annotatedIndexFull[netName] == commons.rede1_positive)
    evalDataset = annotatedIndexFull.loc[mask, :]
elif rede == 1:
    evalDataset = annotatedIndexFull.copy()

# Translate to binary classes
evalDataset[netName] = dutils.translate_labels(evalDataset[netName], netName)
dutils.df_to_csv(evalDataset, binaryDatasetPath)
