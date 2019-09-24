import pandas               as pd
import numpy                as np
import sklearn.metrics      as skm
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils


def check_inside_threshold(output, upper_thresh, lower_thresh):
    ''' Checks if output in (-inf, lower_thresh] U [upper_thresh, +inf) '''
    return np.logical_or( np.greater(output, upper_thresh), np.less(output, lower_thresh))


indexPath   = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images_iteration_1.csv"
outputPaths = Path(dirs.saved_models)/ "results_full_dataset_iteration_0.pickle"

pickleData = utils.load_pickle(outputPaths)
indexDf    = pd.read_csv(indexPath)
indexDf.set_index("FrameHash", drop=False)

# print(pickleData[:20])
# exit()
outputs      = np.stack(pickleData["Outputs"])#[:, 0]
imgHashes    = pickleData["ImgHashes"]
labels       = pickleData["Labels"]
datasetLen   = len(outputs)

## Compute predictions comparing the greater score of the output pair
# predictionsMax = np.argmax(outputs, axis=1)

# # Invert class index for checking --> makes no difference
# # predictionsMax = np.where(predictionsMax == 1, 0, 1)

# accuracyMax    = np.sum(np.equal(labels, predictionsMax))/len(labels)
# print("Measuring max output of the score pair")
# print("Accuracy (max of pair): {:.2f} %".format(accuracyMax*100))

# Keep only target class scores
outputs = outputs[:, 0]
orderedIndex = np.argsort(outputs)

print("\nStatistics before normalization")
print("max : ", np.max(outputs, axis=0))
print("min : ", np.min(outputs, axis=0))
print("mean: ", np.mean(outputs, axis=0))
print("std : ", np.std(outputs, axis=0))

outputs = np.squeeze(utils.normalize_array(outputs))

print("\nStatistics after normalization")
print("max : ", np.max(outputs, axis=0))
print("min : ", np.min(outputs, axis=0))
print("mean: ", np.mean(outputs, axis=0))
print("std : ", np.std(outputs, axis=0))


# Find upper threshold
upperThreshList = np.arange(1., 0., -0.001)
idealUpperThresh = dutils.find_ideal_upper_thresh(outputs, labels, upperThreshList)

# Find lower threshold
lowerThreshList = np.arange(0., 1., 0.001)
idealLowerThresh = dutils.find_ideal_lower_thresh(outputs, labels, lowerThreshList)

indexes = np.arange(datasetLen, dtype=int)
# print(np.greater(outputs, idealUpperThresh))
# print(np.less(outputs, idealLowerThresh))
# exit()
upperClassified = indexes[np.greater(outputs, idealUpperThresh)]
lowerClassified = indexes[np.less(outputs, idealLowerThresh)]
totalClassified = len(upperClassified) + len(lowerClassified)

print("upperClassified: ", len(upperClassified))
print("lowerClassified: ", len(lowerClassified))
print("\nImages automatically labeled: {}/{} = {:.2f} %".format(totalClassified, datasetLen,
                                                              (totalClassified)/datasetLen*100))

# upperThresh = .85
# lowerThresh = .2
# limit = 50

# upperMask = np.greater(outputs, upperThresh)
# lowerMask = np.less(outputs, lowerThresh)

# upperIndexes  = np.arange(datasetLen)[upperMask]
# lowerIndexes  = np.arange(datasetLen)[lowerMask]

# upperLen      = len(upperIndexes)
# lowerLen      = len(lowerIndexes)

# upperSelected = outputs[upperIndexes]
# lowerSelected = outputs[lowerIndexes]

# # upperPreds    = np.where(upperMask, 0, 1)
# # lowerPreds    = np.where(lowerMask, 1, 0)

# upperCorrects = np.sum(np.equal(0, labels[upperIndexes]))
# lowerCorrects = np.sum(np.equal(1, labels[lowerIndexes]))

# upperPrecision = upper_positive_relative_ratio(outputs, labels, upperThresh)
# lowerRecall    = lower_positive_ratio(outputs, labels, lowerThresh)

# # Compute accuracy from a single score: classification threshold as 0.5
# predictions    = np.where(outputs > .5, 0, 1)
# accuracy       = skm.accuracy_score(labels, predictions)


# # ------------
# print("\nAscending order of score and label pairs")
# # High scores should correlate with class index 0 and vice versa
# print("Score\tGround Truth")
# # for i in upperIndexes:
# for i in orderedIndex:
#     print("{:.4f}\t {}".format(outputs[i], labels[i]))
# # # print(len(upperSelected))
# # # print(len(upperIndexes))
# # print("Accuracy (pos > 0.5): {:.2f} %".format(accuracy*100))
# input()

# # for ind in upperIndexes:
# #     imgHash = imgHashes[ind]
# #     img = indexDf.loc[imgHash]
#     title = "output: {:.2f}".format(outputs[ind])
#     dutils.show_image(img, title_string=title)
