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
import libs.commons         as commons
import libs.dataset_utils   as dutils
import models.utils         as mutils
from models.trainer_class   import TrainModel
from libs.index             import IndexManager

if __name__ == "__main__":
    seed = 33
    mutils.set_torch_random_seeds(seed)
    iteration   = 2
    epochs      = 100
    rede        = 1
    batchSize = 64

    unlabelIndexPath  = Path(dirs.iter_folder) / \
                "full_dataset/iteration_{}/unlabeled_images_iteration_{}.csv".format(iteration, iteration)
    savedModelsFolder = Path(dirs.saved_models) / "full_dataset_rede_{}/iteration_{}".format(rede, iteration)
    modelPath         = savedModelsFolder / \
                "full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(epochs, rede, iteration)
    savePath          = savedModelsFolder / \
                "outputs_full_dataset_{}_epochs_rede_{}_iteration_{}.pickle".format(epochs, rede, iteration)


    unlabelIndex = IndexManager(unlabelIndexPath)

    # Drop duplicated files
    unlabelIndex.index = dutils.remove_duplicates(unlabelIndex.index, "FrameHash")

    # Drop missing or corrupt images
    unlabelIndex.index = dutils.check_df_files(unlabelIndex.index, utils.check_empty_file, "FramePath")

    
    imagePathList = unlabelIndex.index["FramePath"].values
    datasetLen    = len(imagePathList)

    print("\nDataset information: ")
    print("\t", datasetLen, "images.")
    
    # ImageNet statistics
    mean    = commons.IMAGENET_MEAN
    std     = commons.IMAGENET_STD 

    # Set transforms
    dataTransforms = mutils.resnet_transforms(mean, std)['val']
    
    # Label list for an unlabeled dataset (bit of a hack? is there a better way?)
    labelList = np.zeros(datasetLen)

    imgLoader = dutils.IndexLoader(imagePathList, batch_size=batchSize,
                                   transform=dataTransforms, label_list=labelList)
    
    # for img, imgHash in imgLoader:
    #     # print(img)
    #     # print(np.shape(img))
    #     print(imgHash)
    #     # break

    # Instantiate trainer object
    trainer = TrainModel(model_path=modelPath)
    trainer.numClasses = 2      # Sloppily set model's number of output units

    # Set model
    trainer.define_model_resnet18(finetune=False, print_summary=True)

    # Perform inference here for testing
    # ------------------
    # img = Image.open(imagePathList[0])
    # img = torch.stack([dataTransforms(img)], dim=0)
    # img = img.to('cuda:0')

    # trainer.model.eval()
    # with torch.set_grad_enabled(False):
    #     output1 = trainer.model(img)
    # print("Op 1: ", output1)
    # dutils.show_inputs(img, output1)

    # with torch.set_grad_enabled(False):
    #     output2 = trainer.model(img)
    # print("Op 2: ", output2)
    # dutils.show_inputs(img, output2)
    # -------------------
    
    outputs, imgHashes, labels = trainer.model_inference(imgLoader)

    outputDf = pd.DataFrame({"Outputs":   outputs,
                             "ImgHashes": imgHashes,
                             "Labels":    labels})

    print(np.shape(outputDf))

    ## Save output to pickle file
    print("\nSaving outputs file to ", savePath)
    outputDf.to_pickle(savePath)

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
