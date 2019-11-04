import torch
import pandas                   as pd
import numpy                    as np
import torch.nn                 as nn
import torchvision.datasets     as datasets
from pathlib                    import Path

import libs.dirs                as dirs
import libs.commons             as commons
import libs.utils               as utils
import libs.dataset_utils       as dutils
import models.utils             as mutils
from models.trainer_class       import TrainModel
from libs.index                 import IndexManager
from libs.vis_functions         import plot_outputs_histogram
from libs.iteration_manager     import SampleImages


if __name__ == "__main__":
    iteration = int(input("Enter iteration number.\n"))
    seed           = 42
    # iteration      = 3
    rede           = 1
    epochs         = 500

    datasetName = "full_dataset_softmax"

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "{}/iteration_{}/".format(datasetName, iteration)

    datasetFolder        = dirs.febe_image_dataset
    previousIterFolder   = get_iter_folder(iteration-1)
    iterFolder           = get_iter_folder(iteration)
    sampledImageFolder   = iterFolder / "sampled_images"
    imageResultsFolder   = Path(dirs.results) / \
              "{}_rede_{}_softmax/iteration_{}".format(datasetName, rede, iteration)

    originalUnlabeledIndexPath = get_iter_folder(0) / "unlabeled_images_iteration_0.csv"
    unlabeledIndexPath    = previousIterFolder / "unlabeled_images_iteration_{}.csv".format(iteration-1)
    sampledIndexPath      = iterFolder / "sampled_images_iteration_{}.csv".format(iteration)
    manualIndexPath       = iterFolder / "manual_annotated_images_iteration_{}.csv".format(iteration)
    prevManualIndexPath   = previousIterFolder / \
        "manual_annotated_images_iteration_{}_train_val_split.csv".format(iteration-1)
    splitIndexPath        = iterFolder / (manualIndexPath.stem + "_train_val_split.csv")
    autoLabelIndexPath    = iterFolder / "automatic_labeled_images_iteration_{}.csv".format(iteration)
    mergedIndexPath       = iterFolder / "final_annotated_images_iteration_{}.csv".format(iteration)

# unlabeledIndexPath  : unlabeled_images contains all the images still not labeled at the end of the iteration. Will be read at the next iteration as a reference
# mergedIndexPath     : final_annotated_images contains all images annotated in an iteration and on the previous iterations
# manualIndexPath     : annotated_images contains the current manually annotated dataset. Include current and previous iterations. Valid from iter > 2
# splitIndexPath      : annotated_images_..._train_val_split contains the annotated images to be used in training this iteration, that is, the manual annotated images from the current and previous iterations.
# autoLabelIndexPath  : automatic_labeled_images contains images annotated automatically in the current iteration

    ## Process manual labels and add missing information
    print("\nSTEP: Process manual labels and add missing information")
    # Add folder path
    def _add_folder_path(path):
        path = datasetFolder / Path(path)
        return str(path)

    # Load model outputs and unlabeled images index
    indexSampled = IndexManager(sampledIndexPath)

    indexSampled.index["FramePath"] = indexSampled.index["imagem"].map(_add_folder_path)

    eTime = indexSampled.compute_frame_hashes(reference_column="FramePath", verbose=True)

    indexSampled.write_index(dest_path=manualIndexPath, make_backup=False, prompt=False)

    ## Merge manual annotated labels from current and previous iterations
    if iteration > 1:
        oldLabels = pd.read_csv(prevManualIndexPath)
        newLabels = pd.read_csv(manualIndexPath)

        # Remove duplicates
        oldLabels = dutils.remove_duplicates(oldLabels, "FrameHash")
        newLabels = dutils.remove_duplicates(newLabels, "FrameHash")

        # Get additional information for newLabels from main unlabeled index
        # TODO: Don't do this again when merging auto and manual annotated indexes
        originalUnlabeledIndex = pd.read_csv(originalUnlabeledIndexPath)
        originalUnlabeledIndex = dutils.remove_duplicates(originalUnlabeledIndex, "FrameHash")

        newLabels = dutils.fill_index_information(originalUnlabeledIndex, newLabels,
                                                 "FrameHash", [ 'rede1', 'rede2', 'rede3'])
        oldLabels = dutils.fill_index_information(originalUnlabeledIndex, oldLabels,
                                                 "FrameHash", [ 'rede1', 'rede2', 'rede3'])

        mergedIndex = pd.concat([newLabels, oldLabels], axis=0, sort=False)
        mergedIndex.to_csv(manualIndexPath, index=False)

    ## Split train and val sets
    print("\nSTEP: Split train and val sets.")
    splitPercentages = [0.8, 0.2]

    # Move images from dataset folder to sampled images
    # Sort images in sampled_images folder to separate class folders
    dutils.move_dataset_to_train(manualIndexPath, sampledImageFolder, path_column="FramePath")
    
    imageIndex = dutils.move_to_class_folders(manualIndexPath, sampledImageFolder, target_net="rede1")
    input("\nDelete unwanted class folders and press Enter to continue.")

    # Split dataset in train and validation sets, sorting them in val and train folders
    splitIndex = dutils.data_folder_split(sampledImageFolder,
                                        splitPercentages, index=imageIndex.copy(), seed=seed)
    splitIndex.to_csv(splitIndexPath, index=False)
