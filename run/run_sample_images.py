from pathlib import Path

import libs.dirs            as dirs
from libs.iteration_manager import SampleImages

seed = 33

# datasetName = "all_datasets_1s"

# Source: image folder
# locale      = dirs.febe_images  # Remote path
locale      = dirs.images       # Local path
# sourcePath  = Path(locale + datasetName)
# destFolder  = Path(locale + "sampled_images_temp2/")

# sourcePath = Path(dirs.dataset) / "compiled_dataset_2019-8-2_16-30-1"
# destFolder  = Path(dirs.iter_folder) / "test_loop/iteration_1/"

# Source: index file
sourcePath = Path(dirs.iter_folder)/ "full_dataset/iteration_2/unlabeled_images_iteration_2.csv"
destFolder = Path(dirs.iter_folder) / "full_dataset/iteration_2/"
# destFolder = Path(dirs.dataset) / "temp/compiled_dataset_temp_index"

sampler = SampleImages(sourcePath, destFolder, seed=seed)
sampler.sample(percentage=0.01)
# print(sampler.imageSourcePaths)
print(sampler.imageSourcePaths.shape)

# sampler.save_to_index()
