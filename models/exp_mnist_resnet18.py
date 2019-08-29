import torchvision.datasets as datasets
from torchvision            import models, transforms
from torch.utils.data       import random_split

import libs.dirs            as dirs
from libs.utils             import *
from models.trainer_class   import TrainModel


if __name__ == "__main__":
    # datasetPath = Path(dirs.dataset) / "torch/hymenoptera_data"
    datasetPath = Path(dirs.dataset) / "torch/mnist"

    # ImageNet statistics
    mean    = [0.485, 0.456, 0.406]
    std     = [0.229, 0.224, 0.225]

    # Idea: compare with data statistics

    dataTransforms = {
        'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
        ])
    }

    # Dataset loaders for train and val sets
    # imageDatasets = {
    #     x: datasets.ImageFolder(str(datasetPath / x), dataTransforms[x]) for x in ['train', 'val']
    # }

    imageDatasets = datasets.MNIST(datasetPath, train=True, transform=None, download=True)
    datasetLen = len(imageDatasets)

    trainPercentage = 0.8
    valPercentage   = 0.2

    trainSize = int(datasetLen*trainPercentage)
    valSize   = int(datasetLen*valPercentage)

    dataset = {}
    dataset['train'], dataset['val'] = random_split(imageDatasets, [trainSize, valSize])
    print(len(dataset['train']))
    print(len(dataset['val']))
    print(type(imageDatasets))



    # Get an image batch of the training set
    # inputs, classes = next(iter(dataloaders['train']))

    # Make a grid and display it
    # imgGrid = torchvision.utils.make_grid(inputs)
    # torch_imshow(imgGrid, mean, std, title=[classNames[x] for x in classes])
    # plt.show()

    # Instantiate trainer object
    # trainer = TrainModel()
    
    # device = torch.device('cuda:0')
    # modelFineTune = trainer.define_model_resnet18(finetune=True)

    # criterion = nn.CrossEntropyLoss()

    # # Set optimizer
    # optimizerFineTune = optim.SGD(modelFineTune.parameters(), lr=0.001, momentum=0.9)

    # # Scheduler for learning rate decay
    # expLrScheduler = lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)

    # # Perform training
    # trainer.load_data(imageDatasets, num_examples_per_batch=4)
    # modelFineTune = trainer.train(modelFineTune, criterion, optimizerFineTune, expLrScheduler, num_epochs=25)
