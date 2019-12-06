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
                "full_dataset_softmax/iteration_{}/unlabeled_images_iteration_{}.csv".format(iteration-1, iteration-1)
savedModelsFolder = Path(dirs.saved_models) / "full_dataset_rede_{}_softmax/iteration_{}".format(rede, iteration)
outputPath   = savedModelsFolder / \
                "outputs_full_dataset_{}_epochs_rede_{}_iteration_{}.pickle".format(epochs, rede, iteration)
newIndexPath = Path(dirs.iter_folder) / \
                "full_dataset/iteration_{}/automatic_labeled_images_iteration_{}.csv".format(iteration, iteration)

idealUpperThresh = 0.8923 # Ratio 99%
idealLowerThresh = 0.0904 # Ratio 1%

indexDf    = pd.read_csv(indexPath)
pickleData = utils.load_pickle(outputPath)


indexDf     = dutils.remove_duplicates(indexDf, "FrameHash")
outputs, imgHashes, _ = dutils.load_outputs_df(outputPath)

outputs = outputs[:, 0]

indexDf.set_index("FrameHash", drop=False, inplace=True)

print("\nAutomatic labeling with upper positive ratio 99%:")
posHashes, negHashes = dutils.automatic_labeling(outputs, imgHashes,
                                                 idealUpperThresh, idealLowerThresh)

newLabeledIndex = dutils.get_classified_index(indexDf, posHashes, negHashes, verbose=False)

# newLabeledIndex.to_csv(newIndexPath, index=False)

imgSavePath = Path(dirs.results) / "histogram_unlabeled_outputs.pdf"
plot_outputs_histogram(outputs, lower_thresh=idealLowerThresh, upper_thresh=idealUpperThresh,
                       title="Unlabeled Outputs Histogram", save_path=imgSavePath, show=True)
