import os
import time
import copy
import torch
import matplotlib
import torchvision
import torch.nn             as nn
import numpy                as np
import torch.optim          as optim
import matplotlib.pyplot    as plt
from pathlib                import Path
from torch.optim            import lr_scheduler
from torchvision            import datasets, models, transforms

import libs.dirs            as dirs
from libs.utils             import *

matplotlib.use("Qt4Agg")


class TrainModel:
    def __init__(self, model, criterion, optimizer, scheduler):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.num_epochs = None
        self.num_batches= None

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.bestAcc          = 0.
        self.bestModelWeights = copy.deepcopy(self.model.state_dict())


    def load_data(self, dataset, num_batches):
        self.dataset     = dataset
        self.num_batches = num_batches
        
        self.datasetSizes = {
            x: len(self.dataset[x]) for x in ['train', 'val']
        }
        
        # Batch generator
        # self.dataloaders = {
        #     x: torch.utils.data.DataLoader(dataset[x], batch_size=self.num_batches,
        #                                    shuffle=True, num_workers=4) for x in ['train', 'val']
        # }
        self.dataloaders = {}
        for x in ['train', 'val']:
            self.dataloaders[x] = torch.utils.data.DataLoader(self.dataset[x], batch_size=self.num_batches,
                                                              shuffle=True, num_workers=4)

        self.classNames = dataset['train'].classes
        
        return self.dataloaders


    def train(self, num_epochs=25):
        self.num_epochs = num_epochs
        self.start      = time.time()
        # Training loop
        for self.epoch in range(self.num_epochs):
            print("\n-----------------------")
            print("Epoch {}/{}".format(self.epoch+1, self.num_epochs))

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()   # Set model to training mode
                else:
                    self.model.eval()    # Set model to evaluate mode

                self.runningLoss     = 0.
                self.runningCorrects = 0

                # Iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    self.inputs = inputs.to(device)
                    self.labels = labels.to(device)

                    # Reset gradients
                    self.optimizer.zero_grad()

                    # Forward propagation
                    trainPhase = (phase == 'train')
                    with torch.set_grad_enabled(trainPhase):
                        self.outputs = self.model(self.inputs)
                        _, self.predictions = torch.max(self.outputs, 1)
                        self.loss = self.criterion(self.outputs, self.labels)

                        # Perform backpropagation only in train phase
                        if trainPhase:
                            self.loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()
                    
                    # Get statistics
                    self.runningLoss += self.loss.item() * self.inputs.size(0)
                    self.runningCorrects += torch.sum(self.predictions == self.labels.data)
                
                self.epochLoss = self.runningLoss/ self.datasetSizes[phase]
                self.epochAcc = self.runningCorrects.double() / self.datasetSizes[phase]

                print("{} Phase\n\tLoss: {:.4f}\n\tAcc : {:.4f}".format(
                                            phase, self.epochLoss, self.epochAcc))

                # Save model if there is an improvement in evaluation metric
                if phase == 'val' and self.epochAcc > self.bestAcc:
                    self.bestAcc = self.epochAcc
                    self.bestModelWeights = copy.deepcopy(self.model.state_dict())

        self.elapsedTime = time.time() - self.start
        print("\nTraining completed. Elapsed time: {:.0f}:{:.0f}".format(
                                            self.elapsedTime / 60, self.elapsedTime % 60))
        # print("\nTraining completed. Elapsed time: ", get_time_string(self.elapsedTime))
        print("Best val accuracy: {:.4f}".format(self.bestAcc))

        # Load best model weights
        self.model.load_state_dict(self.bestModelWeights)
        return self.model


def torch_imshow(gridInput, mean, std, title=None):
    gridInput = gridInput.numpy().transpose((1,2,0))

    gridInput = std*gridInput + mean
    gridInput = np.clip(gridInput, 0, 1)

    ax = plt.imshow(gridInput)
    plt.title(title)
    # plt.pause(0.01)
    # plt.imsave("../images/testgrid.png", gridInput)

if __name__ == "__main__":
    datasetPath = Path(dirs.dataset) / "torch/hymenoptera_data"

    mean    = [0.485, 0.456, 0.406]
    std     = [0.229, 0.224, 0.225]

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
    imageDatasets = {
        x: datasets.ImageFolder(str(datasetPath / x), dataTransforms[x]) for x in ['train', 'val']
    }


    # Get an image batch of the training set
    # inputs, classes = next(iter(dataloaders['train']))

    # Make a grid and display it
    # imgGrid = torchvision.utils.make_grid(inputs)
    # torch_imshow(imgGrid, mean, std, title=[classNames[x] for x in classes])
    # plt.show()

    device = torch.device('cuda:0')
    # Load pre-trained model
    modelFineTune    = models.resnet18(pretrained=True)
    numFeatures      = modelFineTune.fc.in_features
    modelFineTune.fc = nn.Linear(numFeatures, 2)
    modelFineTune.to(device)

    criterion = nn.CrossEntropyLoss()

    # Set optimizer
    optimizerFineTune = optim.SGD(modelFineTune.parameters(), lr=0.001, momentum=0.9)

    # Scheduler for learning rate decay
    expLrScheduler = lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)

    # Perform training
    trainer = TrainModel(modelFineTune, criterion, optimizerFineTune, expLrScheduler)
    trainer.load_data(imageDatasets, num_batches=4)
    modelFineTune = trainer.train(num_epochs=25)
