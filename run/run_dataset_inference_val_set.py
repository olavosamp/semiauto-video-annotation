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
from models.trainer_class   import TrainModel
from libs.index             import IndexManager


if __name__ == "__main__":
    seed = 33
    dutils.set_torch_random_seeds(seed)

    datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images/val/"
    savePath    = Path(dirs.saved_models)/ "outputs_full_dataset_validation_iteration_0_rede1.pickle"
    modelPath   = Path(dirs.saved_models)/ "full_dataset_no_finetune_1000_epochs_rede1.pt"
    
    batchSize = 64

    # ImageNet statistics
    # No need to normalize pixel range from [0, 255] to [0, 1] because
    # ToTensor transform already does that
    mean    = [0.485, 0.456, 0.406]#/255
    std     = [0.229, 0.224, 0.225]#/255
    
    dataTransforms = transforms.Compose([
                        transforms.Resize(256), # Pq 256?
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

    # Get list of image paths from dataset folder
    dataset = datasets.ImageFolder(str(datasetPath),
                                    transform=dataTransforms,
                                    is_valid_file=utils.check_empty_file)
    imageTupleList  = dataset.imgs
    datasetLen      = len(imageTupleList)
    labelList       = dataset.targets
    # labelList       = list(range(datasetLen)) # Test with sequential labels
    
    imagePathList  = np.array(dataset.imgs)[:, 0]

    print("\nDataset information: ")
    print("\t", datasetLen, "images.")
    print("\nClasses: ")
    for key in dataset.class_to_idx.keys():
        print("\t{}: {}".format(dataset.class_to_idx[key], key))

    imgLoader = dutils.IndexLoader(imagePathList, batch_size=batchSize, transform=dataTransforms, label_list=labelList)
    
    # for img, imgHash in imgLoader:
    #     # print(img)
    #     # print(np.shape(img))
    #     print(imgHash)
    #     # break

    # Instantiate trainer object
    trainer = TrainModel(model_path=modelPath)
    trainer.numClasses = 2

    # Set model
    trainer.define_model_resnet18(finetune=False, print_summary=True)
    # exit()
    
    # Perform inference
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


    # predictions = np.argmax(np.array(outputs), axis=1)
    # accuracy = np.equal(predictions, labels).sum()/datasetLen
    

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