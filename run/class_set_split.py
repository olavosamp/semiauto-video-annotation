import pandas as pd
import numpy  as np

from pathlib                  import Path
import libs.dirs              as dirs
import libs.commons           as commons
import libs.utils             as utils
import libs.dataset_utils     as dutils

iterPath        = Path(dirs.iter_folder) / "full_dataset/iteration_2/"
indexPath       = iterPath / "annotated_images_iteration_2.csv"
imageFolderPath = iterPath / "sampled_images"
datasetFolder   = dirs.dataset / "all_datasets_1s"
savePath        = indexPath.parent / (indexPath.stem + "_train_val_split.csv")
seed = 42

splitPercentages = [0.8, 0.2]

# Move images from dataset folder to sampled images
imageIndex = pd.read_csv(indexPath)
def add_folder_and_copy(x): return utils.copy_files(datasetFolder / Path(x), imageFolderPath / Path(x))
successes = np.sum(imageIndex["FramePath"].map(add_folder_and_copy))

print("{}/{} files moved.".format(successes, len(imageIndex["FramePath"])))


imageIndex = dutils.move_to_class_folders(indexPath, imageFolderPath, target_net="rede1")
input("\nDelete unwanted class folders and press Enter to continue.")

# Split dataset in train and validation sets
otherIndex = dutils.data_folder_split(imageFolderPath,
                                      splitPercentages, index=imageIndex.copy(), seed=seed)
print(otherIndex.head())
otherIndex.to_csv(savePath, index=False)
