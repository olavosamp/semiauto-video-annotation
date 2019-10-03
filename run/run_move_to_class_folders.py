from pathlib                  import Path
import libs.dirs              as dirs
import libs.commons           as commons
import libs.utils             as utils
import libs.dataset_utils     as dutils

iterPath  = Path(dirs.iter_folder) / "full_dataset/iteration_0/"
indexPath = iterPath / "olavo_uniformsampling_4676_corrections.csv"

splitPercentages = [0.85, 0.15]

imageIndex = dutils.move_to_class_folders(indexPath, imageFolder="sampled_images")

savePath = indexPath.parent / (indexPath.stem + "_train_val_split.csv")
imageIndex.to_csv(savePath, index=False)

# input("\nDelete unwanted class folders and press Enter to continue.")
## Split dataset in train and validation sets
# dutils.data_folder_split(iterPath / "sampled_images", splitPercentages)
