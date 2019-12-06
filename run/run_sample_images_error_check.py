from pathlib import Path

import libs.dirs            as dirs
import libs.dataset_utils   as dutils
import libs.commons         as commons
from libs.iteration_manager import SampleImages

seed       = 33
percentage = 0.01

rede      = int(input("Enter net number.\n"))

if rede == 3:
    target_class = dutils.get_input_target_class(commons.rede3_classes)
    datasetName  = "full_dataset_rede_{}_{}".format(rede, target_class.lower())
else:
    datasetName  = "full_dataset_rede_{}".format(rede)

sets = ['manual', 'automatic']
# Sample images from given sets
for currentSet in sets:
    source = Path(dirs.iter_folder) / "{}/final_{}_images_{}.csv".format(datasetName,
                                                                     currentSet, datasetName)
    dest      = Path(dirs.images) / "samples_error_check/{}/{}".format(datasetName, currentSet)
    save_path = Path(dirs.images) / "samples_error_check/{}/{}_check_index.csv".format(datasetName, currentSet)

    sampler = SampleImages(source, dest, seed=seed)
    sampler.sample(percentage=percentage)
    print(sampler.imageSourcePaths.shape)

    sampler.save_to_index(save_path)

# print(sampler.imageSourcePaths)
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
