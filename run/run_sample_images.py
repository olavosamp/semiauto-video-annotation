from pathlib import Path

import libs.dirs            as dirs
from libs.iteration_manager import SampleImages

seed       = 33
percentage = 0.01
# Manual images
source     = Path(dirs.iter_folder) / "full_dataset_softmax/final_manual_images.csv"
dest       = Path(dirs.images) / "samples_error_check/manual"

# datasetName = "all_datasets_1s"

## Source from an image folder
## locale      = dirs.febe_images  # Remote path
# locale      = dirs.images       # Local path
# sourcePath  = Path(locale + datasetName)
# destFolder  = Path(locale + "sampled_images_temp2/")

# sourcePath = Path(dirs.dataset) / "compiled_dataset_2019-8-2_16-30-1"
# destFolder  = Path(dirs.iter_folder) / "test_loop/iteration_1/"

# Source from an index file
# sourcePath = Path(dirs.iter_folder)/ "full_dataset/iteration_2/unlabeled_images_iteration_2.csv"
# destFolder = Path(dirs.iter_folder) / "full_dataset/iteration_2/"
# destFolder = Path(dirs.dataset) / "temp/compiled_dataset_temp_index"

sampler = SampleImages(source, dest, seed=seed)
sampler.sample(percentage=percentage)
# print(sampler.imageSourcePaths)
print(sampler.imageSourcePaths.shape)

# sampler.save_to_index()
