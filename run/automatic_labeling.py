import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
from tqdm                   import tqdm
from PIL                    import Image
from pathlib                import Path
from copy                   import copy

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from libs.vis_functions     import plot_outputs_histogram

outputPath      = Path(dirs.saved_models)/ "outputs_full_dataset_iteration_0_1000_epochs_rede1.pickle"
indexPath       = Path(dirs.iter_folder) / "full_dataset/iteration_0/unlabeled_images_iteration_1.csv"
newIndexPath    = Path(dirs.iter_folder) / "full_dataset/iteration_0/automatic_labeled_images_iteration_1.csv"

indexDf    = pd.read_csv(indexPath)
pickleData = utils.load_pickle(outputPath)

indexDf.set_index("FrameHash", drop=False, inplace=True)
pickleData.set_index("ImgHashes", drop=False, inplace=True)

print("\nFound and removed {} duplicated entries in unlabeled images Index.".format(
                                                                    np.sum(indexDf.index.duplicated())))
indexDf    = indexDf[~indexDf.index.duplicated()]
pickleData = pickleData[~pickleData.index.duplicated()]
# TODO: Find out why there are 9k duplicated images in unlabeled images index
pickleData.reset_index(drop=True, inplace=True)

outputs    = np.stack(pickleData["Outputs"])[:, 0]
outputs    = utils.normalize_array(outputs)
datasetLen = len(outputs)

idealLowerThresh = 0.3690 # Ratio 1%
idealUpperThresh = 0.5191 # Ratio 99%
print("\nAutomatic labeling with upper positive ratio 99%:")
upperClassified, lowerClassified = dutils.automatic_labeling(outputs, idealUpperThresh, idealLowerThresh)

newLabels = np.concatenate([upperClassified, lowerClassified])
newHashes = pickleData.loc[newLabels, "ImgHashes"].values

newLabeledIndex = indexDf.reindex(labels=newHashes, axis=0, copy=True)

print(newLabeledIndex.shape)
print("outputs:       ", datasetLen)
print("new labels:    ", len(newHashes))
print("new labels df: ", newLabeledIndex.shape)
print(len(newLabeledIndex)/datasetLen*100)

newLabeledIndex.to_csv(newIndexPath, index=False)

imgSavePath = Path(dirs.results) / "histogram_unlabeled_outputs.pdf"
plot_outputs_histogram(outputs, lower_thresh=idealLowerThresh, upper_thresh=idealUpperThresh,
                       title="Unlabeled Outputs Histogram", save_path=imgSavePath, show=False)
