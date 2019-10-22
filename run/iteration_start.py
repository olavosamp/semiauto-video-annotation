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


if __name__ == "__main__":
    seed           = 42
    iteration      = 1
    rede           = 1
    epochs         = 1000
    trainBatchSize = 256
    inferBatchSize = 64

    unlabeledSetFolder  = Path(dirs.dataset) / "all_datasets_1s"
    iterPath            = Path(dirs.iter_folder) / "full_dataset/iteration_{}/".format(iteration)
    imageFolderPath     = iterPath / "sampled_images"
    savedModelsFolder   = Path(dirs.saved_models) / "full_dataset_rede_{}/iteration_{}".format(rede, iteration)
    imageResultsFolder  = Path(dirs.results) / \
              "full_dataset_history_no_finetune_{}_epochs_rede_{}_iteration_{}".format(epochs, rede, iteration)
    
    indexPath           = iterPath / "annotated_images_iteration_{}.csv".format(iteration)
    splitSavePath       = iterPath / (indexPath.stem + "_train_val_split.csv")
    datasetValPath      = imageFolderPath / "val/"

    modelPath      = savedModelsFolder / \
                    "full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(epochs, rede, iteration)
    historyPath    = savedModelsFolder / \
        "history_full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(epochs, rede, iteration)

    outputPath     = savedModelsFolder / \
                    "outputs_full_dataset_validation_rede_{}_iteration_{}.pickle".format(iteration, rede)
    resultsFolder = imageResultsFolder / "histogram_output_rede_{}_softmax".format(rede)

    # Debug all file paths
    print("\nunlabeledSetFolder: {}\n{} ".format( Path(unlabeledSetFolder).is_dir(), unlabeledSetFolder))
    print("\niterPath: {}\n{} ".format( Path(iterPath).is_dir(), iterPath))
    print("\nimageFolderPath: {}\n{} ".format( Path(imageFolderPath).is_dir(), imageFolderPath))
    print("\nsavedModelsFolder: {}\n{} ".format( Path(savedModelsFolder).is_dir(), savedModelsFolder))
    print("\nimageResultsFolder: {}\n{} ".format( Path(imageResultsFolder).is_dir(), imageResultsFolder))
    print("\ndatasetValPath: {}\n{} ".format( Path(datasetValPath).is_dir(), datasetValPath))

    print("\nindexPath: {}\n{} ".format( Path(indexPath).is_file(), indexPath))
    print("\nsplitSavePath: {}\n{} ".format( Path(splitSavePath).is_file(), splitSavePath))
    print("\nmodelPath: {}\n{} ".format( Path(modelPath).is_file(), modelPath))
    print("\nhistoryPath: {}\n{} ".format( Path(historyPath).is_file(), historyPath))
    print("\noutputPath: {}\n{} ".format( Path(outputPath).is_file(), outputPath))
    exit()

    ## Split train and val sets
    splitPercentages = [0.8, 0.2]

    # Move images from dataset folder to sampled images
    # Sort images in sampled_images folder to separate class folders
    dutils.move_dataset_to_train(indexPath, unlabeledSetFolder, path_column="FramePath")
    imageIndex = dutils.move_to_class_folders(indexPath, imageFolderPath, target_net="rede1")
    # input("\nDelete unwanted class folders and press Enter to continue.")

    # Split dataset in train and validation sets, sorting them in val and train folders
    splitIndex = dutils.data_folder_split(imageFolderPath,
                                        splitPercentages, index=imageIndex.copy(), seed=seed)
    splitIndex.to_csv(splitSavePath, index=False)

    ## Train model
    # ImageNet statistics
    mean    = commons.IMAGENET_MEAN
    std     = commons.IMAGENET_STD 

    # Set transforms
    dataTransforms = mutils.resnet_transforms(mean, std)

    history, modelFineTune = mutils.train_network(imageFolderPath, dataTransforms, epochs=25, batch_size=64,
                                            model_path=modelPath,
                                            history_path=historyPath)

    # TODO: Plot train history here. Encapsulate scripts to functions and put here


    ## Dataset Inference on Validation set to find thresholds
    mutils.set_torch_random_seeds(seed)

    # Perform inference on validation set and save outputs to file
    outputDf = mutils.dataset_inference_val(datasetValPath, dataTransforms['val'], modelPath,
                                            outputPath, batch_size=inferBatchSize)

    valOutputs, imgHashes, labels = dutils.load_outputs_df(outputPath)

    # Compute decision thresholds
    upperThresh, lowerThresh = dutils.compute_thresholds(valOutputs,
                                                                   labels,
                                                                   upper_ratio=0.99,
                                                                   lower_ratio=0.01,
                                                                   resolution=0.0001,#resolution='max',
                                                                   verbose=True)

    # Plot validation outputs histogram
    valOutputs = valOutputs[:, 0]
    plot_outputs_histogram(valOutputs, labels, lowerThresh, upperThresh, show=False,
                           save_path= resultsFolder /\
                           "histogram_val_set_output_thresholds_softmax_iteration_{}_test.png".format(iteration))
    
