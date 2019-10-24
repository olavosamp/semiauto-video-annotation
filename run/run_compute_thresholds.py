import pandas               as pd
import numpy                as np
import sklearn.metrics      as skm
from pathlib                import Path
import matplotlib.pyplot       as plt

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from libs.vis_functions     import plot_outputs_histogram


valOutputPath = Path(dirs.saved_models) / "full_dataset_rede_1/iteration_1/outputs_full_dataset_validation_rede_1_iteration_1.pickle"
# pickleData = utils.load_pickle(valOutputPath)

# outputs      = np.stack(pickleData["Outputs"])
# imgHashes    = pickleData["ImgHashes"]
# labels       = pickleData["Labels"]
valOutputs, imgHashes, labels = dutils.load_outputs_df(valOutputPath)
print(valOutputs.shape)
idealUpperThresh, idealLowerThresh = dutils.compute_thresholds(valOutputs,
                                                               labels,
                                                               upper_ratio=0.99,
                                                               lower_ratio=0.01,
                                                               resolution=0.0001,
                                                               val_indexes=imgHashes)

# Plot outputs histogram
# valOutputs = valOutputs[:, 0]
# valOutputs = np.squeeze(utils.normalize_array(outputs))
# plot_outputs_histogram(outputs, labels, idealLowerThresh, idealUpperThresh,
#                        save_path=Path(dirs.results)/"histogram_val_set_output_thresholds.png")

## Compute predictions comparing the greater score of the output pair
# predictionsMax = np.argmax(outputs, axis=1)

# # Invert class index for checking --> makes no difference
# # predictionsMax = np.where(predictionsMax == 1, 0, 1)

# accuracyMax    = np.sum(np.equal(labels, predictionsMax))/len(labels)
# print("Measuring max output of the score pair")
# print("Accuracy (max of pair): {:.2f} %".format(accuracyMax*100))

# print("\nStatistics before normalization")
# print("max : ", np.max(outputs, axis=0))
# print("min : ", np.min(outputs, axis=0))
# print("mean: ", np.mean(outputs, axis=0))
# print("std : ", np.std(outputs, axis=0))

# # Keep only target class scores
# # orderedIndex = np.argsort(outputs)

# print("\nStatistics after normalization")
# print("max : ", np.max(outputs, axis=0))
# print("min : ", np.min(outputs, axis=0))
# print("mean: ", np.mean(outputs, axis=0))
# print("std : ", np.std(outputs, axis=0))


# datasetLen      = len(outputs)
# indexes         = np.arange(datasetLen, dtype=int)
# upperClassified = indexes[np.greater(outputs, idealUpperThresh)]
# lowerClassified = indexes[np.less(outputs, idealLowerThresh)]
# totalClassified = len(upperClassified) + len(lowerClassified)

# print("\nIdeal Upper Threshold: ", idealUpperThresh)
# print("Ideal Lower Threshold: ", idealLowerThresh)

# print("\nResults in Validation set:")
# print("upperClassified: ", len(upperClassified))
# print("lowerClassified: ", len(lowerClassified))
# print("\nImages automatically labeled: {}/{} = {:.2f} %".format(totalClassified, datasetLen,
#                                                             (totalClassified)/datasetLen*100))


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
