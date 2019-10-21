import os
import torch
import math
import random
import numpy                as np
import pandas               as pd
import torchvision.datasets as datasets
from PIL                    import Image
from pathlib                import Path
from torchvision            import transforms
from torch.utils.data       import DataLoader

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
import models.utils         as mutils
from models.trainer_class   import TrainModel
from libs.index             import IndexManager
from libs.vis_functions     import plot_outputs_histogram


if __name__ == "__main__":
    seed = 33
    mutils.set_torch_random_seeds(seed)
    iteration   = 1
    epochs      = 1000
    rede        = 1

    datasetPath       = Path(dirs.iter_folder) / \
                        "full_dataset/iteration_{}/sampled_images/val/".format(iteration)
    savedModelsFolder = Path(dirs.saved_models) / "full_dataset_rede_{}/iteration_{}".format(rede, iteration)
    savePath          = savedModelsFolder / \
                        "outputs_full_dataset_validation_rede_{}_iteration_{}.pickle".format(rede, iteration)
    modelPath         = savedModelsFolder / \
                    "full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(epochs, rede, iteration)
    batchSize = 64

    # Define transforms
    # ImageNet statistics
    mean    = [0.485, 0.456, 0.406]#/255
    std     = [0.229, 0.224, 0.225]#/255
    
    # ToTensor transform normalizes pixel range to [0, 1]
    dataTransforms = transforms.Compose([
                        transforms.Resize(256), # Pq 256?
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

    ## Perform inference on validation set and save outputs to file
    # outputDf = mutils.dataset_inference_val(datasetPath, dataTransforms, modelPath, savePath, batch_size=batchSize)

    # outputPath = Path(dirs.saved_models) / "outputs_full_dataset_validation_iteration_1_rede1.pickle"
    outputPath = savePath
    outputs, imgHashes, labels = dutils.load_outputs_df(outputPath)
    
    # print(np.shape(outputs))
    # print(np.sum(outputs, axis=1)[:20])
    # print(outputs[:20, :])
    # exit()

    idealUpperThresh, idealLowerThresh = dutils.compute_thresholds(outputs,
                                                                   labels,
                                                                   upper_ratio=0.99,
                                                                   lower_ratio=0.01,
                                                                   resolution=0.0001,#resolution='max',
                                                                   verbose=True)
    exit()
    # Plot outputs histogram
    outputs = outputs[:, 0]
    saveFolder = Path(dirs.results) / "histogram_output_rede_{}_softmax".format(rede)
    plot_outputs_histogram(outputs, labels, idealLowerThresh, idealUpperThresh,
                           save_path= saveFolder /\
                           "histogram_val_set_output_thresholds_softmax_iteration_{}_test.png".format(iteration))
    # plot_outputs_histogram(outputs, labels,
    #                        save_path= saveFolder /\
    #                        "histogram_val_set_output_thresholds_softmax_iteration_{}_test.png".format(iteration))

    # # Find upper threshold
    # upperThreshList = np.arange(1., 0., -0.001)
    # idealUpperThresh = dutils.find_ideal_upper_thresh(outputs, labels, upperThreshList)

    # # Find lower threshold
    # lowerThreshList = np.arange(0., 1., 0.001)
    # idealLowerThresh = dutils.find_ideal_lower_thresh(outputs, labels, lowerThreshList)

    # outputs = np.stack(outputs)[:, 0]
    # outputs = utils.normalize_array(outputs)

    # idealUpperThresh = 0.392
    # idealLowerThresh = 0.224

    # indexes = np.arange(datasetLen, dtype=int)
    # upperClassified = indexes[np.greater(outputs, idealUpperThresh)]
    # lowerClassified = indexes[np.less(outputs, idealLowerThresh)]
    # totalClassified = len(upperClassified) + len(lowerClassified)

    # print("upperClassified: ", len(upperClassified))
    # print("lowerClassified: ", len(lowerClassified))
    # print("\nImages automatically labeled: {}/{} = {:.2f} %".format(totalClassified, datasetLen,
    #                                                             (totalClassified)/datasetLen*100))
