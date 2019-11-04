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
from libs.vis_functions         import plot_outputs_histogram
from libs.iteration_manager     import SampleImages


if __name__ == "__main__":
    iteration = int(input("Enter iteration number.\n"))
    seed           = 42
    # iteration      = 3
    rede           = 1
    epochs         = 500
    inferBatchSize = 64

    datasetName = "full_dataset_softmax"

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "{}/iteration_{}/".format(datasetName, iteration)

    remoteDatasetFolder  = Path(dirs.dataset) / "all_datasets_1s"
    previousIterFolder   = get_iter_folder(iteration-1)
    iterFolder           = get_iter_folder(iteration)
    nextIterFolder       = get_iter_folder(iteration+1)
    sampledImageFolder   = iterFolder / "sampled_images"
    savedModelsFolder    = Path(dirs.saved_models) / \
        "{}_rede_{}/iteration_{}".format(datasetName, rede, iteration)
    valSetFolder         = sampledImageFolder / "val/"
    imageResultsFolder   = Path(dirs.results) / \
        "{}_rede_{}/iteration_{}".format(datasetName, rede, iteration)

    modelPath            = savedModelsFolder / \
        "{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(datasetName, epochs, rede, iteration)
    historyPath          = savedModelsFolder / \
        "history_{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(datasetName, epochs, rede, iteration)
    valOutputPath        = savedModelsFolder / \
        "outputs_{}_validation_rede_{}_iteration_{}.pickle".format(datasetName, rede, iteration)
    fullOutputPath       = savedModelsFolder / \
        "outputs_{}_{}_epochs_rede_{}_iteration_{}.pickle".format(datasetName, epochs, rede, iteration)

    originalUnlabeledIndexPath = get_iter_folder(0) / "unlabeled_images_iteration_0.csv"
    unlabeledIndexPath      = previousIterFolder / "unlabeled_images_iteration_{}.csv".format(iteration-1)
    sampledIndexPath        = iterFolder / "sampled_images_iteration_{}.csv".format(iteration)
    manualIndexPath         = iterFolder / "manual_annotated_images_iteration_{}.csv".format(iteration)
    splitIndexPath          = iterFolder / (manualIndexPath.stem + "_train_val_split.csv")
    autoLabelIndexPath      = iterFolder / "automatic_labeled_images_iteration_{}.csv".format(iteration)
    mergedIndexPath         = iterFolder / "final_annotated_images_iteration_{}.csv".format(iteration)
    previousMergedIndexPath = previousIterFolder / "final_annotated_images_iteration_{}.csv".format(iteration-1)
    newUnlabeledIndexPath   = iterFolder / "unlabeled_images_iteration_{}.csv".format(iteration)

    unlabelHistogramPath = imageResultsFolder / "histogram_unlabeled_outputs_iteration_{}.pdf".format(iteration)
    valHistogramPath     = imageResultsFolder / "histogram_validation_outputs_iteration_{}.pdf".format(iteration)

    ## Dataset Inference on Validation set to find thresholds
    print("\nSTEP: Perform inference on val set.")
    # ImageNet statistics
    mean    = commons.IMAGENET_MEAN
    std     = commons.IMAGENET_STD 

    # Set transforms
    dataTransforms = mutils.resnet_transforms(mean, std)

    # Perform inference on validation set and save outputs to file
    outputDf = mutils.dataset_inference_val(valSetFolder, dataTransforms['val'], modelPath,
                                            valOutputPath, batch_size=inferBatchSize, seed=seed)

    # Compute decision thresholds
    print("\nSTEP: Compute decision thresholds.")
    valOutputs, imgHashes, labels = dutils.load_outputs_df(valOutputPath)
    upperThresh, lowerThresh = dutils.compute_thresholds(valOutputs,
                                                        labels,
                                                        upper_ratio=0.99,
                                                        lower_ratio=0.01,
                                                        resolution=0.0001,#resolution='max',
                                                        val_indexes=imgHashes)

    # Plot validation outputs histogram
    valOutputs = valOutputs[:, 0]
    plot_outputs_histogram(valOutputs, labels, lowerThresh, upperThresh, show=False,
                           save_path = valHistogramPath, log=True)

    ## Perform inference on entire unlabeled dataset
    print("\nSTEP: Perform inference on entire dataset.")
    mutils.dataset_inference_unlabeled(unlabeledIndexPath, dataTransforms['val'], modelPath, fullOutputPath,
                        batch_size=inferBatchSize, seed=seed, verbose=True)
    
    # upperThresh = 0.8923 # Ratio 99%
    # lowerThresh = 0.0904 # Ratio 1%

    print("\nUsing thresholds:\nUpper: {:.4f}\nLower: {:.4f}".format(upperThresh, lowerThresh))

    ## Perform automatic labeling
    print("\nSTEP: Automatic labeling.")
    unlabeledIndex  = pd.read_csv(unlabeledIndexPath)
    pickleData      = utils.load_pickle(fullOutputPath)

    unlabeledIndex        = dutils.remove_duplicates(unlabeledIndex, "FrameHash")
    outputs, imgHashes, _ = dutils.load_outputs_df(fullOutputPath)
    outputs = outputs[:, 0]

    print("\nAutomatic labeling with upper positive ratio 99%:")
    posHashes, negHashes = dutils.automatic_labeling(outputs, imgHashes,
                                                    upperThresh, lowerThresh)

    newLabeledIndex = dutils.get_classified_index(unlabeledIndex, posHashes, negHashes,
                                                    index_col="FrameHash", verbose=False)
    newLabeledIndex.to_csv(autoLabelIndexPath, index=False)

    plot_outputs_histogram(outputs, lower_thresh=lowerThresh, upper_thresh=upperThresh,
                        title="Unlabeled Outputs Histogram", save_path=unlabelHistogramPath,
                        log=True, show=False)

    ## Merge labeled sets
    print("\nMerge auto and manual labeled sets.")
    # Merge annotated images of the current iteration: manual and auto
    sampledIndex           = pd.read_csv(sampledIndexPath)
    autoIndex              = pd.read_csv(autoLabelIndexPath)
    originalUnlabeledIndex = pd.read_csv(originalUnlabeledIndexPath)
    
    originalUnlabeledIndex = dutils.remove_duplicates(originalUnlabeledIndex, "FrameHash")

    # Add FrameHash column
    sampledIndex["FrameHash"] = utils.compute_file_hash_list(sampledIndex["imagem"].values,
                                                            folder=dirs.febe_image_dataset)

    # Get missing information from original Unlabeled index
    sampledIndex = dutils.fill_index_information(originalUnlabeledIndex, sampledIndex,
                                            "FrameHash", [ 'rede1', 'rede2', 'rede3'])

    sampledIndex = dutils.remove_duplicates(sampledIndex, "FrameHash")
    autoIndex    = dutils.remove_duplicates(autoIndex, "FrameHash")

    mergedIndex = dutils.merge_manual_auto_sets(sampledIndex, autoIndex)
    print(mergedIndex.shape)

    mergedIndex.to_csv(mergedIndexPath, index=False)


    ## Create unlabeled set for next iteration
    print("\nCreate new unlabeled set.")
    mergedPathList = [get_iter_folder(x) / \
        "final_annotated_images_iteration_{}.csv".format(x) for x in range(1, iteration+1)]
    mergedIndexList = [pd.read_csv(x) for x in mergedPathList]
    originalUnlabeledIndex  = pd.read_csv(originalUnlabeledIndexPath)
    # mergedIndex             = pd.read_csv(mergedIndexPath)
    # previousMergedIndex     = pd.read_csv(previousMergedIndexPath)
    print(originalUnlabeledIndex.index.shape)

    originalUnlabeledIndex = dutils.remove_duplicates(originalUnlabeledIndex, "FrameHash")

    # print("Shape final_annotations_iter_{}: {}".format(iteration, mergedIndex.shape))
    # print("Shape final_annotations_iter_{}: {}".format(iteration-1, previousMergedIndex.shape))

    # allAnnotations = pd.concat([previousMergedIndex, mergedIndex], axis=0, sort=False)
    allAnnotations = pd.concat(mergedIndexList, axis=0, sort=False)
    
    allAnnotations = dutils.remove_duplicates(allAnnotations, "FrameHash")
    print("Duplicated elements in ogUnlabelIndex and allAnnottations.")
    print(originalUnlabeledIndex.index.duplicated().sum())
    print(allAnnotations.index.duplicated().sum())

    newIndex = dutils.index_complement(originalUnlabeledIndex, allAnnotations, "FrameHash")
    print(newIndex.shape)

    # dirs.create_folder(newUnlabeledIndexPath.parent)
    newIndex.to_csv(newUnlabeledIndexPath, index=False)

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
