import pandas               as pd
import numpy                as np
import sklearn.metrics      as skm
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils


def check_inside_threshold(output, upper_thresh, lower_thresh):
    # if output >= upper_thresh or output <= lower_thresh:
    #     return True
    # else:
    #     return False
    return np.logical_or( np.greater(output, upper_thresh), np.less(output, lower_thresh))


def normalize_array(array):
    maxVal = np.max(array)
    minVal = np.min(array)
    dif = np.abs(maxVal - minVal)

    return (array - minVal)/(dif)

def compute_upper_lower_metrics(outputs, labels, upperThresh, lowerThresh):
    upperMask = np.greater(outputs, upperThresh)
    lowerMask = np.less(outputs, lowerThresh)

    upperIndexes  = np.arange(datasetLen)[upperMask]
    lowerIndexes  = np.arange(datasetLen)[lowerMask]

    upperPrecision = skm.precision_score(labels[upperIndexes],
                                         np.zeros(len(upperIndexes), dtype=int), pos_label=0)
    
    lowerRecall    = skm.recall_score(labels[lowerIndexes],
                                      np.ones(len(lowerIndexes), dtype=int), pos_label=0)

    return upperPrecision, lowerRecall



indexPath   = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images_iteration_1.csv"
outputPaths = Path(dirs.saved_models)/ "results_full_dataset_iteration_0.pickle"

pickleData = utils.load_pickle(outputPaths)
indexDf    = pd.read_csv(indexPath)
indexDf.set_index("FrameHash", drop=False)

# print("Scores for two class outputs")

print(pickleData.shape)
outputs   = np.stack(pickleData[0, :])[:, 0]
imgHashes = np.array(pickleData[1, :])
labels = []
# labels    = np.array(pickleData[2, :])

datasetLen = len(outputs)

for elem in pickleData[2, :]:
    labels.extend(elem)
labels    = np.array(labels)

# print(outputs.shape)
# print(imgHashes.shape)
# print(labels.shape)
# print(outputs[:20])
# exit()

print("\nStatistics before normalization")
print("max : ", np.max(outputs))
print("min : ", np.min(outputs))
print("mean: ", np.mean(outputs))
print("std : ", np.std(outputs))

outputs = normalize_array(outputs)

print("\nStatistics after normalization")
print("max : ", np.max(outputs))
print("min : ", np.min(outputs))
print("mean: ", np.mean(outputs))
print("std : ", np.std(outputs))
print()
# input()

upperThresh = .8
lowerThresh = .2
limit = 50

# threshMask = check_inside_threshold(outputs, 0.9, 0.2)

upperMask = np.greater(outputs, upperThresh)
lowerMask = np.less(outputs, lowerThresh)

upperIndexes  = np.arange(datasetLen)[upperMask]
lowerIndexes  = np.arange(datasetLen)[lowerMask]

upperLen      = len(upperIndexes)
lowerLen      = len(lowerIndexes)

upperSelected = outputs[upperIndexes]
lowerSelected = outputs[lowerIndexes]

# upperPreds    = np.where(upperMask, 0, 1)
# lowerPreds    = np.where(lowerMask, 1, 0)

upperCorrects = np.sum(np.equal(0, labels[upperIndexes]))
lowerCorrects = np.sum(np.equal(1, labels[lowerIndexes]))

upperPrecision= skm.precision_score(labels[upperIndexes], np.zeros(upperLen, dtype=int), pos_label=0)
lowerRecall   = skm.recall_score(labels[lowerIndexes], np.ones(lowerLen, dtype=int), pos_label=0)


# for i in range(upperLen):
#     print

print("\nupperCorrects:")
print("Labels:     ", labels[upperIndexes])
print("Predictions:", np.zeros(upperLen, dtype=int))
# print(upperSelected, labels[upperIndexes])
print("Corrects : ", upperCorrects)
print("Precision: ", upperPrecision)
print("\nlowerCorrects:")
print("Labels:     ", labels[lowerIndexes])
print("Predictions:", np.ones(lowerLen, dtype=int))
# print(lowerSelected, labels[lowerIndexes])
print("Corrects: ", lowerCorrects)
print("Recall:   ", lowerRecall)


# print("\nUpper Threshold: ", upperThresh)
# # print(upperSelected[:limit])
# print(upperIndexes[:limit])

# print("Score\tPrediction\tGround Truth")
# for i in upperIndexes:
#     print("{:.4f}\t {} \t\t {}".format(outputs[i], 0, labels[i]))
# print(len(upperSelected))
# print(len(upperIndexes))

# for ind in upperIndexes:
#     imgHash = imgHashes[ind]
#     img = indexDf.loc[imgHash]
#     title = "output: {:.2f}".format(outputs[ind])
#     dutils.show_image(img, title_string=title)

# print("\nLower Threshold: ", lowerThresh)
# print(lowerSelected[:limit])
# for i in range(limit):
#     if lowerMask[i]:
#         print("{:.4f}: {}".format(outputs[i], lowerMask[i]))
