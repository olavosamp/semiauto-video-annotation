import torch
import numpy                as np
import torchvision.datasets as datasets
import torch.nn             as nn
import torch.optim          as optim
from torchvision            import models, transforms
from torch.utils.data       import random_split
from pathlib                import Path

import libs.dirs            as dirs
from models.trainer_class   import MnistTrainer


def tensor_to_3_channel_greyscale(tensor):
    return torch.cat((tensor, tensor, tensor), 0)

def rgb_to_greyscale(img):
    return 0.299 *img[0] + 0.587 *img[1] + 0.114 * img[2]

if __name__ == "__main__":
    datasetPath = Path(dirs.dataset) / "torch/mnist"
    numImgBatch = 128

    modelPath   = dirs.saved_models + "test_mnist_resnet18.pt"
    historyPath = dirs.saved_models + "test_mnist_resnet18_history.pickle"

    # ImageNet statistics
    # No need to normalize pixel range from [0, 255] to [0, 1] because
    # ToTensor transform already does that
    mean    = [0.485, 0.456, 0.406]#/255
    std     = [0.229, 0.224, 0.225]#/255

    # mean    = [rgb_to_greyscale(mean)] * 3
    # std     = [rgb_to_greyscale(std)] * 3

    # Idea: compare with actual data statistics
    
    # Set transforms
    dataTransforms = {
        'train': transforms.Compose([
                    transforms.Resize(224),
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(tensor_to_3_channel_greyscale),
                    transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(tensor_to_3_channel_greyscale),
                    transforms.Normalize(mean, std),
        ])
    }

    dataset = {}
    dataset['train']  = datasets.MNIST(datasetPath, train=True, transform=dataTransforms['train'],
                                        download=True)
    dataset['val']    = datasets.MNIST(datasetPath, train=False, transform=dataTransforms['val'],
                                        download=True)
    
    # print(np.shape(dataset['train'].__getitem__(0)[0]))
    # print(np.shape(dataset['train'].__getitem__(0)[1]))
    # print(dataset['train'].targets.size())
    # print(dataset['train'].targets.max())
    # print(dataset['train'].targets.min())
    # print(dataset['train'].__getitem__(0).size())
    print(dataset['train'])
    # input()
    # exit()

    # Instantiate trainer object
    trainer = MnistTrainer()

    # Perform training
    trainer.load_data(dataset, num_examples_per_batch=numImgBatch)

    # Set resnet18 model
    modelFineTune = trainer.define_model_resnet18(finetune=True)

    # Set training parameters
    criterion = nn.CrossEntropyLoss()

    optimizerFineTune = optim.Adam(modelFineTune.parameters())

    expLrScheduler = optim.lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)

    # Train model
    modelFineTune = trainer.train(modelFineTune, criterion, optimizerFineTune, expLrScheduler, num_epochs=25)
    history = trainer.save_history(historyPath)

    # Save model
    torch.save(modelFineTune.state_dict(), modelPath)