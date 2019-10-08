import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
from PIL                    import Image
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
from libs.vis_functions     import plot_outputs_histogram

outputPath      = Path(dirs.saved_models)/ "outputs_full_dataset_iteration_0_1000_epochs_rede1.pickle"
indexPath       = Path(dirs.iter_folder) / "full_dataset/iteration_0/unlabeled_images_iteration_1.csv"
indexDf         = pd.read_csv(indexPath)
indexDf.set_index("FrameHash", drop=False)

pickleData = utils.load_pickle(outputPath)

# Select only unlabeled data
pickleData.set_index("ImgHashes", drop=False)
pickleData = pickleData.loc[indexDf.index]

outputs    = np.stack(pickleData["Outputs"])[:, 0]
outputs    = utils.normalize_array(outputs)
datasetLen = len(outputs)

idealLowerThresh = 0.3690 # Ratio 1%

# idealUpperThresh = xx # Ratio 95%
# print("\nAutomatic labeling with upper positive ratio 95%:")
# _, _ = dutils.automatic_labeling(outputs, idealUpperThresh, idealLowerThresh)

idealUpperThresh = 0.5191 # Ratio 99%
print("\nAutomatic labeling with upper positive ratio 99%:")
upperClassified, lowerClassified = dutils.automatic_labeling(outputs, idealUpperThresh, idealLowerThresh)

newLabels = np.concatenate([upperClassified, lowerClassified])
# newLabels = 
print(newLabels[:20])
newLabeledIndex = indexDf.loc[newLabels, :]
print(newLabeledIndex.shape)

imgSavePath = Path(dirs.results) / "histogram_unlabeled_outputs.pdf"
plot_outputs_histogram(outputs, lower_thresh=idealLowerThresh, upper_thresh=idealUpperThresh,
                       title="Unlabeled Outputs Histogram", save_path=imgSavePath, show=False)
