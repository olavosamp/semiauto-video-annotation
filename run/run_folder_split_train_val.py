import libs.dirs            as dirs
from pathlib                import Path
from libs.dataset_utils     import data_folder_split


datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images/"

# Split datasets in train and validation sets
trainPercentage = 0.8
valPercentage   = 0.2

# Should be run only once to split images in train and val folders
data_folder_split(datasetPath, [trainPercentage, valPercentage])
