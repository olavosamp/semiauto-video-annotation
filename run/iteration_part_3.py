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
    seed           = np.random.randint(0, 100)
    rede           = 2
    epochs         = 150
    inferBatchSize = 64
    datasetName = "full_dataset_rede_{}".format(rede)

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "{}/iteration_{}/".format(datasetName, iteration)

    remoteDatasetFolder  = Path(dirs.dataset) / "all_datasets_1s"
    previousIterFolder   = get_iter_folder(iteration-1)
    iterFolder           = get_iter_folder(iteration)
    nextIterFolder       = get_iter_folder(iteration+1)
    sampledImageFolder   = iterFolder / "sampled_images"
    savedModelsFolder    = Path(dirs.saved_models) / \
        "{}/iteration_{}".format(datasetName, iteration)
    valSetFolder         = sampledImageFolder / "val/"
    imageResultsFolder   = Path(dirs.results) / \
        "{}/iteration_{}".format(datasetName, iteration)

    modelPath            = savedModelsFolder / \
        "{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(datasetName, epochs, rede, iteration)
    historyPath          = savedModelsFolder / \
        "history_{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(datasetName, epochs, rede, iteration)
    valOutputPath        = savedModelsFolder / \
        "outputs_{}_validation_rede_{}_iteration_{}.pickle".format(datasetName, rede, iteration)
    fullOutputPath       = savedModelsFolder / \
        "outputs_{}_{}_epochs_rede_{}_iteration_{}.pickle".format(datasetName, epochs, rede, iteration)

    # originalUnlabeledIndexPath = get_iter_folder(0) / "reference_images.csv"
    originalUnlabeledIndexPath = get_iter_folder(0) / "unlabeled_images_iteration_0.csv"
    unlabeledIndexPath         = previousIterFolder / "unlabeled_images_iteration_{}.csv".format(iteration-1)
    sampledIndexPath           = iterFolder / "sampled_images_iteration_{}.csv".format(iteration)
    manualIndexPath            = iterFolder / "manual_annotated_images_iteration_{}.csv".format(iteration)
    splitIndexPath             = iterFolder / (manualIndexPath.stem + "_train_val_split.csv")
    autoLabelIndexPath         = iterFolder / "automatic_labeled_images_iteration_{}.csv".format(iteration)
    mergedIndexPath            = iterFolder / "final_annotated_images_iteration_{}.csv".format(iteration)
    previousMergedIndexPath    = previousIterFolder / "final_annotated_images_iteration_{}.csv".format(iteration-1)
    unlabelNoManualPath        = iterFolder / "unlabeled_no_manual_iteration_{}.csv".format(iteration)
    newUnlabeledIndexPath      = iterFolder / "unlabeled_images_iteration_{}.csv".format(iteration)

    unlabelHistogramPath = imageResultsFolder / "histogram_unlabeled_outputs_iteration_{}.pdf".format(iteration)
    valHistogramPath     = imageResultsFolder / "histogram_validation_outputs_iteration_{}.pdf".format(iteration)

    reportPath           = iterFolder/"report_iteration_{}.txt".format(iteration)
    seedLogPath          = iterFolder / "seeds.txt"

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
    plot_outputs_histogram(valOutputs[:, 0], labels, lowerThresh, upperThresh, show=False,
                           save_path = valHistogramPath, log=True)

    ## Perform inference on entire unlabeled dataset
    # Get unlabeled set without manual_annotated_images
    originalUnlabeledIndex = pd.read_csv(originalUnlabeledIndexPath)
    # splitIndex = pd.read_csv(splitIndexPath)

    # TODO: Maybe compute unlabeled_index_no_manual earlier: at part_2 after manual annotation 
    # Do now: unlabeledNoManualIndex = complement(unlabeled_it_0, [final_annot_it_1, manual_annot_split_it_2])
    annotatedSoFarIndex  = dutils.merge_indexes([previousMergedIndexPath, splitIndexPath], "FrameHash")
    unlabelNoManualIndex = dutils.index_complement(originalUnlabeledIndex, annotatedSoFarIndex, "FrameHash")
    unlabelNoManualIndex.to_csv(unlabelNoManualPath, index=False)

    # If outputs file already exist, skip inference
    print("\nSTEP: Perform inference on remaining unlabeled set.")
    if not(fullOutputPath.is_file()):
        mutils.dataset_inference_unlabeled(unlabelNoManualPath, dataTransforms['val'], modelPath,
                            fullOutputPath, batch_size=inferBatchSize, seed=seed, verbose=True)
    else:
        print("Output file already exists: {}\nSkipping inference.".format(fullOutputPath))

    print("\nUsing thresholds:\nUpper: {:.4f}\nLower: {:.4f}".format(upperThresh, lowerThresh))

    ## Perform automatic labeling
    print("\nSTEP: Automatic labeling.")
    unlabeledNoManualIndex = pd.read_csv(unlabelNoManualPath)
    pickleData             = utils.load_pickle(fullOutputPath)

    outputs, imgHashes, _  = dutils.load_outputs_df(fullOutputPath)
    outputs = outputs[:, 0]

    print("\nAutomatic labeling with upper positive ratio 99%:")
    autoIndex = dutils.automatic_labeling(outputs, imgHashes, unlabeledNoManualIndex, upperThresh,
                                                     lowerThresh, rede)
    autoIndex.to_csv(autoLabelIndexPath, index=False)

    plot_outputs_histogram(outputs, lower_thresh=lowerThresh, upper_thresh=upperThresh,
                        title="Unlabeled Outputs Histogram", save_path=unlabelHistogramPath,
                        log=True, show=False)

    ## Merge labeled sets
    print("\nMerge auto and manual labeled sets.")
    # Merge annotated images of the current iteration: manual and auto
    # Must use sampledIndex because manualIndex also has manual images from previous iterations
    sampledIndex           = pd.read_csv(sampledIndexPath)
    autoIndex              = pd.read_csv(autoLabelIndexPath)
    originalUnlabeledIndex = pd.read_csv(originalUnlabeledIndexPath)

    # TODO: Encapsulate sampledImages processing in function
    # Add FrameHash column
    if "imagem" in sampledIndex.columns:
        fileList = sampledIndex["imagem"].values
    elif "FrameName" in sampledIndex.columns:
        fileList = sampledIndex["FrameName"].values
    else:
        raise KeyError("DataFrame doesn't have a known image path column.")
    
    sampledIndex["FrameHash"] = utils.compute_file_hash_list(fileList, folder=dirs.febe_image_dataset)

    # Get missing information from original Unlabeled index
    sampledIndex = dutils.fill_index_information(originalUnlabeledIndex, sampledIndex,
                                            "FrameHash", [ 'rede1', 'rede2', 'rede3'])

    sampledIndex = dutils.remove_duplicates(sampledIndex, "FrameHash")
    autoIndex    = dutils.remove_duplicates(autoIndex, "FrameHash")

    # TODO: Use merge_indexes in merge_manual_auto_sets
    mergedIndex = dutils.merge_manual_auto_sets(sampledIndex, autoIndex)
    print(mergedIndex.shape)

    mergedIndex.to_csv(mergedIndexPath, index=False)


    ## Create unlabeled set for next iteration
    # TODO: Encapsulate this section in function
    print("\nCreate new unlabeled set.")
    mergedPathList = [get_iter_folder(x) / \
        "final_annotated_images_iteration_{}.csv".format(x) for x in range(1, iteration+1)]
    mergedIndexList = [pd.read_csv(x) for x in mergedPathList]
    originalUnlabeledIndex  = pd.read_csv(originalUnlabeledIndexPath)

    # print("Shape final_annotations_iter_{}: {}".format(iteration, mergedIndex.shape))
    # print("Shape final_annotations_iter_{}: {}".format(iteration-1, previousMergedIndex.shape))

    allAnnotations = pd.concat(mergedIndexList, axis=0, sort=False)

    allAnnotations = dutils.remove_duplicates(allAnnotations, "FrameHash")
    print("Duplicated elements in final_annotated_images.")
    print(allAnnotations.index.duplicated().sum())

    newIndex = dutils.index_complement(originalUnlabeledIndex, allAnnotations, "FrameHash")

    dirs.create_folder(newUnlabeledIndexPath.parent)
    newIndex.to_csv(newUnlabeledIndexPath, index=False)

    dutils.make_report(reportPath, sampledIndexPath, manualIndexPath, autoLabelIndexPath,
                       unlabeledIndexPath, None, rede=rede)

    # Save sample seed
    dutils.save_seed_log(seedLogPath, seed, "inference")
