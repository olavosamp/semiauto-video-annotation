import numpy                as np
import sklearn              as sk
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


outputPaths = Path(dirs.saved_models)/ "results_full_dataset_iteration_0.pickle"

pickleData = utils.load_pickle(outputPaths)

# print("Scores for two class outputs")

outputs   = np.array(pickleData[0])[:, 0]
imgHashes = np.array(pickleData[1])
labels    = []
for elem in pickleData[2]:
    labels.extend(elem)
labels    = np.array(labels)

# print(outputs.shape)
# print(imgHashes.shape)
# print(labels.shape)
# print(labels)
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

upperThreshMask = np.greater(outputs, upperThresh)
lowerThreshMask = np.less(outputs, lowerThresh)

# threshMask = check_inside_threshold(outputs, 0.9, 0.2)

limit = 50
print("\nupperThreshMask")
for i in range(limit):
    print("{:.4f}: {}".format(outputs[i], upperThreshMask[i]))

print("\nlowerThreshMask")
for i in range(limit):
    print("{:.4f}: {}".format(outputs[i], lowerThreshMask[i]))