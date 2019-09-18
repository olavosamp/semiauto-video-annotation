import os
import torch
import numpy                as np
import torchvision.datasets as datasets
import torch.nn             as nn
import torch.optim          as optim
from PIL                    import Image
from pathlib                import Path
from torchvision            import models, transforms
from torch.utils.data       import DataLoader

import libs.dirs            as dirs
import libs.utils           as utils
from libs.dataset_utils     import data_folder_split
from models.trainer_class   import TrainModel


def check_file_size(path):
    return os.path.getsize(path) > 0

# class IndexLoader(DataLoader):
#     def __init__()
#     def __iter__(self):

class IndexLoader():
    '''
        Iterator to load and transform an image and its file hash.

        Returns
    '''
    def __init__(self, imagePathList, label_list=None, batch_size=1, transform=None):
        self.imagePathList  = imagePathList
        self.batch_size     = batch_size
        self.transform      = transform
        self.label_list     = label_list
        
        self.current_index  = 0
        self.datasetLen     = len(self.imagePathList)

        if self.label_list != None:
            assert len(self.label_list) == self.datasetLen, "Image path and label lists must be of same size."

        # TODO: (maybe) add default Compose transform with ToTensor
        # and Transpose to return a Tensor image with shape (channel, width, height)
        # if self.transform != None:
        #     self.transform = transforms.ToTensor()

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= self.datasetLen:
            raise StopIteration

        imgHash = utils.file_hash(self.imagePathList[self.current_index])
        img = Image.open(self.imagePathList[self.current_index])
        if self.transform:
            img = self.transform(img)

        if self.label_list != None:
            label = self.label_list[self.current_index]
        
        self.current_index += 1
        if self.label_list == None:
            return img, imgHash
        else:
            return img, imgHash, label


if __name__ == "__main__":
    # datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images/"
    datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_0/sampled_images_unsorted/"
    savePath    = Path(dirs.saved_models)/ "results_full_dataset_iteration_0.pickle"
    
    batchSize = 32

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


    dataset = datasets.ImageFolder(str(datasetPath), transform=dataTransforms, is_valid_file=check_file_size)
    # print(dataset.)

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batchSize,
                                            shuffle=False, num_workers=4)

    imageTupleList = dataset.imgs
    imagePathList  = np.array(dataset.imgs)[:, 0]
    # print(np.shape(imagePathList))
    # print(imagePathList[:20])

    datasetLen = len(imageTupleList)
    # iterImg = zip(range(datasetLen)[:20], imageTupleList[:20])

    imgLoader = IndexLoader(imagePathList, transform=None)
    for img, imgHash in imgLoader:
        print(np.shape(img))
        print(imgHash)
        input()
    exit()
    # for imgPath, label in imageTupleList:
    #     imgHash = utils.file_hash(imgPath)
    #     img     = Image.open(imgPath)
    #     img     = dataTransforms(img)
    #     print(img)
    #     print(img.size())
    #     print(imgHash)

    # Instantiate trainer object
    trainer = TrainModel()
    trainer.numClasses = 2

    # Perform training
    # inputDataset = datasets.ImageFolder(datasetPath, transform=dataTransforms['val'])

    modelFineTune = trainer.define_model_resnet18(finetune=False)
    outputs, predictions = trainer.model_inference(dataloader)

    print(predictions[:20])
    print(outputs[:20])

    print("\nSaving outputs file to ", savePath)
    utils.save_pickle(outputs, savePath)
