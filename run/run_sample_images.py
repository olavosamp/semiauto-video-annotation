from pathlib import Path

import libs.dirs            as dirs
from libs.iteration_manager import SampleImages

datasetName = "all_datasets_1s"
# locale      = dirs.febe_images  # Remote path
locale      = dirs.images       # Local path
sourcePath  = Path(locale + datasetName)
destFolder  = Path(locale + "sampled_images_temp2/")

sampler = SampleImages(sourcePath, destFolder)
sampler.sample()
print(sampler.imageSourcePaths)
print(sampler.imageSourcePaths.shape)

sampler.save_to_index()