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
    seed           = 42
    iteration      = 2
    rede           = 1
    epochs         = 1000
    trainBatchSize = 256
    inferBatchSize = 64

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "full_dataset_softmax/iteration_{}/".format(iteration)

    remoteDatasetFolder  = Path(dirs.dataset) / "all_datasets_1s"
    previousIterFolder   = get_iter_folder(iteration-1)
    iterFolder           = get_iter_folder(iteration)
    nextIterFolder       = get_iter_folder(iteration+1)
    sampledImageFolder   = iterFolder / "sampled_images"
    savedModelsFolder    = Path(dirs.saved_models) / "full_dataset_rede_{}/iteration_{}".format(rede, iteration)
    valSetFolder         = sampledImageFolder / "val/"
    imageResultsFolder   = Path(dirs.results) / \
              "full_dataset_rede_{}_softmax/iteration_{}".format(rede, iteration)

    modelPath            = savedModelsFolder / \
        "full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(epochs, rede, iteration)
    historyPath          = savedModelsFolder / \
        "history_full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(epochs, rede, iteration)
    valOutputPath        = savedModelsFolder / \
        "outputs_full_dataset_validation_rede_{}_iteration_{}.pickle".format(iteration, rede)
    fullOutputPath       = savedModelsFolder / \
        "outputs_full_dataset_{}_epochs_rede_{}_iteration_{}.pickle".format(epochs, rede, iteration)

    originalUnlabeledIndexPath = get_iter_folder(0) / "unlabeled_images_iteration_0.csv"
    unlabeledIndexPath    = previousIterFolder / "unlabeled_images_iteration_{}.csv".format(iteration-1)
    sampledIndexPath      = iterFolder / "sampled_images_iteration_{}.csv".format(iteration)
    manualIndexPath       = iterFolder / "manual_annotated_images_iteration_{}.csv".format(iteration)
    prevManualIndexPath   = previousIterFolder / \
        "manual_annotated_images_iteration_{}_train_val_split.csv".format(iteration-1)
    splitIndexPath        = iterFolder / (manualIndexPath.stem + "_train_val_split.csv")
    autoLabelIndexPath    = iterFolder / "automatic_labeled_images_iteration_{}.csv".format(iteration)
    mergedIndexPath       = iterFolder / "final_annotated_images_iteration_{}.csv".format(iteration)
    newUnlabeledIndexPath = iterFolder / "unlabeled_images_iteration_{}.csv".format(iteration)

    unlabelHistogramPath = imageResultsFolder / "histogram_unlabeled_outputs_iteration_{}.pdf".format(iteration)
    valHistogramPath     = imageResultsFolder / "histogram_validation_outputs_iteration_{}.pdf".format(iteration)

# unlabeledIndexPath  : unlabeled_images contains all the images still not labeled at the end of the iteration. Will be read at the next iteration as a reference
# mergedIndexPath     : final_annotated_images contains all images annotated in an iteration and on the previous iterations
# manualIndexPath     : annotated_images contains the current manually annotated dataset. Include current and previous iterations. Valid from iter > 2
# splitIndexPath      : annotated_images_..._train_val_split contains the annotated images to be used in training this iteration, that is, the manual annotated images from the current and previous iterations.
# autoLabelIndexPath  : automatic_labeled_images contains images annotated automatically in the current iteration

    ## Process manual labels and create new unlabeled set
    # Add folder path
    def _add_folder_path(path):
        path = sampledImageFolder / Path(path)
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

        # Add folder path
        def _add_folder_path(path):
            path = sampledImageFolder / Path(path)
            return str(path)

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

    # ## Train model
    # MOVED TO iteration_train

    # TODO: Move to iteration_part_3
    # ## Dataset Inference on Validation set to find thresholds
    # mutils.set_torch_random_seeds(seed)

    # # Perform inference on validation set and save outputs to file
    # outputDf = mutils.dataset_inference_val(valSetFolder, dataTransforms['val'], modelPath,
    #                                         valOutputPath, batch_size=inferBatchSize)

    # # Compute decision thresholds
    # valOutputs, imgHashes, labels = dutils.load_outputs_df(valOutputPath)
    # upperThresh, lowerThresh = dutils.compute_thresholds(valOutputs,
    #                                                                labels,
    #                                                                upper_ratio=0.99,
    #                                                                lower_ratio=0.01,
    #                                                                resolution=0.0001,#resolution='max',
    #                                                                val_indexes=imgHashes)

    # # Plot validation outputs histogram
    # valOutputs = valOutputs[:, 0]
    # plot_outputs_histogram(valOutputs, labels, lowerThresh, upperThresh, show=False,
    #                        save_path = valHistogramPath)

    # ## Perform inference on entire unlabeled dataset
    # # TODO

    # upperThresh = 0.8923 # Ratio 99%
    # lowerThresh = 0.0904 # Ratio 1%

    # ## Perform automatic labeling
    # unlabeledIndex  = pd.read_csv(unlabeledIndexPath)
    # pickleData      = utils.load_pickle(fullOutputPath)

    # unlabeledIndex        = dutils.remove_duplicates(unlabeledIndex, "FrameHash")
    # outputs, imgHashes, _ = dutils.load_outputs_df(fullOutputPath)
    # outputs = outputs[:, 0]

    # print("\nAutomatic labeling with upper positive ratio 99%:")
    # posHashes, negHashes = dutils.automatic_labeling(outputs, imgHashes,
    #                                                 upperThresh, lowerThresh)

    # newLabeledIndex = dutils.get_classified_index(unlabeledIndex, posHashes, negHashes,
    #                                                 index_col="FrameHash", verbose=False)

    # # newLabeledIndex.to_csv(autoLabelIndexPath, index=False)

    # plot_outputs_histogram(outputs, lower_thresh=lowerThresh, upper_thresh=upperThresh,
    #                     title="Unlabeled Outputs Histogram", save_path=unlabelHistogramPath, show=False)

    # ## Merge labeled sets
    # manualIndex = pd.read_csv(manualIndexPath)
    # autoIndex   = pd.read_csv(autoLabelIndexPath)

    # manualIndex = dutils.remove_duplicates(manualIndex, "FrameHash")
    # autoIndex   = dutils.remove_duplicates(autoIndex, "FrameHash")

    # # TODO: Do this as the second iteration step
    # unlabeledIndex = pd.read_csv(unlabeledIndexPath)
    # unlabeledIndex = dutils.remove_duplicates(unlabeledIndex, "FrameHash")

    # # Get additional information for manualIndex from main unlabeled index
    # manualIndex = dutils.fill_index_information(unlabeledIndex, manualIndex,
    #                                             "FrameHash", ["rede1", "rede2", "rede3", "set"])
    # # print(manualIndex.head())
    # print(manualIndex.shape)

    # mergedIndex = dutils.merge_manual_auto_sets(manualIndex, autoIndex)
    # # print(mergedIndex.head())
    # print(mergedIndex.shape)

    # mergedIndex.to_csv(mergedIndexPath, index=False)

    # ## Create unlabeled set for next iteration
    # indexUnlabel = pd.read_csv(unlabeledIndexPath)
    # indexSampled = pd.read_csv(mergedIndexPath)
    # print(indexUnlabel.index.shape)

    # indexUnlabel = dutils.remove_duplicates(indexUnlabel, "FrameHash")

    # # indexUnlabel.set_index("FrameHash", drop=False, inplace=True)
    # # indexSampled.set_index("FrameHash", drop=False, inplace=True)

    # print(indexUnlabel.index.duplicated().sum())
    # print(indexSampled.index.duplicated().sum())

    # newIndex = dutils.index_complement(indexUnlabel, indexSampled, "FrameHash")
    # print(newIndex.shape)

    # # dirs.create_folder(newUnlabeledIndexPath.parent)
    # newIndex.to_csv(newUnlabeledIndexPath, index=False)

    # # Debug all file paths
    # print("\nremoteDatasetFolder: {}\n{} ".format( Path(remoteDatasetFolder).is_dir(), remoteDatasetFolder))
    # print("\niterFolder: {}\n{} ".format( Path(iterFolder).is_dir(), iterFolder))
    # print("\nsampledImageFolder: {}\n{} ".format( Path(sampledImageFolder).is_dir(), sampledImageFolder))
    # print("\nsavedModelsFolder: {}\n{} ".format( Path(savedModelsFolder).is_dir(), savedModelsFolder))
    # print("\nimageResultsFolder: {}\n{} ".format( Path(imageResultsFolder).is_dir(), imageResultsFolder))
    # print("\vVaSetlFolder: {}\n{} ".format( Path(valSetFolder).is_dir(), valSetFolder))

    # print("\nmanualIndexPath: {}\n{} ".format( Path(manualIndexPath).is_file(), manualIndexPath))
    # print("\nsplitIndexPath: {}\n{} ".format( Path(splitIndexPath).is_file(), splitIndexPath))
    # print("\nmodelPath: {}\n{} ".format( Path(modelPath).is_file(), modelPath))
    # print("\nhistoryPath: {}\n{} ".format( Path(historyPath).is_file(), historyPath))
    # print("\nvalOutputPath: {}\n{} ".format( Path(valOutputPath).is_file(), valOutputPath))
    # exit()
