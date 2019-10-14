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
datasetFolder   = Path(dirs.dataset) / "all_datasets_1s"
savePath        = indexPath.parent / (indexPath.stem + "_train_val_split.csv")
seed = 42

splitPercentages = [0.8, 0.2]

def move_dataset_to_train(index_path, dataset_folder, path_column="FramePath", verbose=True):
    ''' Move images from dataset folder to sampled images'''
    # def _add_folder_and_copy(x): return utils.copy_files(dataset_folder / Path(x), imageFolderPath / Path(x))
    def _add_folder_and_copy(x): return utils.copy_files(Path(x), imageFolderPath / Path(x).name)
    index = pd.read_csv(index_path)

    successes = np.sum(index[path_column].map(_add_folder_and_copy))
    if verbose:
        print("{}/{} files moved.".format(successes, len(index[path_column])))
    return successes

# Move images from dataset folder to sampled images
move_dataset_to_train(indexPath, datasetFolder, path_column="FramePath")

# Sort images in sampled_images folder to separate class folders
imageIndex = dutils.move_to_class_folders(indexPath, imageFolderPath, target_net="rede1")
input("\nDelete unwanted class folders and press Enter to continue.")

# Split dataset in train and validation sets, sorting them in val and train folders
otherIndex = dutils.data_folder_split(imageFolderPath,
                                      splitPercentages, index=imageIndex.copy(), seed=seed)
print(otherIndex.head())
otherIndex.to_csv(savePath, index=False)
