import numpy                as np
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from copy                   import copy

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
import libs.commons         as commons
from libs.vis_functions     import plot_outputs_histogram

iteration   = 1
epochs      = 1000
rede        = 1

indexPath    = Path(dirs.iter_folder) / \
                "full_dataset/iteration_{}/unlabeled_images_iteration_{}.csv".format(iteration, iteration)
outputPath   = Path(dirs.saved_models) / \
                "outputs_full_dataset_iteration_{}_{}_epochs_rede{}.pickle".format(iteration, epochs, rede)
newIndexPath = Path(dirs.iter_folder) / \
                "full_dataset/iteration_{}/automatic_labeled_images_iteration_{}.csv".format(iteration, iteration)

idealUpperThresh = 0.8923 # Ratio 99.99%
idealLowerThresh = 0.0904 # Ratio 0.01%

indexDf    = pd.read_csv(indexPath)
pickleData = utils.load_pickle(outputPath)

positiveLabel = commons.rede1_positive
negativeLabel = commons.rede1_negative

indexDf     = dutils.remove_duplicates(indexDf, "FrameHash")
outputs, imgHashes, _ = dutils.load_outputs_df(outputPath)
outputs = outputs[:, 0]
# pickleData  = dutils.remove_duplicates(pickleData, "ImgHashes")

indexDf.set_index("FrameHash", drop=False, inplace=True)
# pickleData.set_index("ImgHashes", drop=False, inplace=True)

# outputs    = np.stack(pickleData["Outputs"])[:, 0]
# outputs    = utils.normalize_array(outputs)
datasetLen = len(outputs)

print("\nAutomatic labeling with upper positive ratio 99%:")
upperClassified, lowerClassified = dutils.automatic_labeling(outputs, idealUpperThresh, idealLowerThresh)

# newLabels = np.concatenate([upperClassified, lowerClassified])
# newHashes = pickleData.loc[newLabels, "ImgHashes"].values
# newLabeledIndex = indexDf.reindex(labels=newHashes, axis=0, copy=True)

posHashes = imgHashes[upperClassified]
negHashes = imgHashes[lowerClassified]

newPositives = indexDf.reindex(labels=posHashes, axis=0, copy=True)
newNegatives = indexDf.reindex(labels=negHashes, axis=0, copy=True)

lenPositives = len(newPositives)
lenNegatives = len(newNegatives)
# Set positive and negative class labels
newPositives["rede1"] = [positiveLabel]*lenPositives
newNegatives["rede1"] = [negativeLabel]*lenNegatives

newLabeledIndex = pd.concat([newPositives, newNegatives], axis=0, sort=False)


print(newLabeledIndex.shape)
print("outputs:          ", datasetLen)
print("\nnew pos labels:   ", lenPositives)
print("new neg labels:   ", lenNegatives)
print("total new labels: ", lenPositives+lenNegatives)
print("new labels df:    ", newLabeledIndex.shape)
print(len(newLabeledIndex)/datasetLen*100)

# newLabeledIndex.to_csv(newIndexPath, index=False)

imgSavePath = Path(dirs.results) / "histogram_unlabeled_outputs.pdf"
plot_outputs_histogram(outputs, lower_thresh=idealLowerThresh, upper_thresh=idealUpperThresh,
                       title="Unlabeled Outputs Histogram", save_path=imgSavePath, show=True)
