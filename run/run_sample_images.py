from pathlib import Path

import libs.dirs            as dirs
from libs.iteration_manager import SampleImages

datasetName = "all_datasets_1s"

# Source: image folder
# locale      = dirs.febe_images  # Remote path
locale      = dirs.images       # Local path
# sourcePath  = Path(locale + datasetName)
# destFolder  = Path(locale + "sampled_images_temp2/")
sourcePath = Path(dirs.dataset) / "compiled_dataset_2019-8-2_16-30-1"
destFolder  = Path(dirs.iter_folder) / "test_loop/iteration_0/"

# # Source: index file
# sourcePath = Path("./index/") / "main_index_2019-7-25_15-1-8.csv"
# destFolder = Path(dirs.dataset) / "temp/compiled_dataset_temp_index"

sampler = SampleImages(sourcePath, destFolder)
sampler.sample(percentage=0.2)
# print(sampler.imageSourcePaths)
# print(sampler.imageSourcePaths.shape)

sampler.save_to_index()
