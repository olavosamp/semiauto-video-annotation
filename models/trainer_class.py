import time
import copy
import torch
import matplotlib
import torchvision
import torch.nn             as nn
from pathlib                import Path
from torchvision            import models

import libs.dirs            as dirs
from libs.utils             import *


class TrainModel:
    def __init__(self):
        self.model                  = None
        self.criterion              = None
        self.optimizer              = None
        self.scheduler              = None
        self.num_epochs             = None
        self.finetune               = None
        self.num_examples_per_batch = None

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.bestAcc          = 0.
        self.bestModelWeights = None


    def load_data(self, dataset, num_examples_per_batch=4):
        '''
            Generate dataloaders for input dataset.

            dataset: dict-like dataset object
                Dataset dict-like object that must contain 'train' and 'val' keys.

            num_examples_per_batch: int
                Number of examples per batch.
        '''
        self.dataset = dataset
        self.num_examples_per_batch = num_examples_per_batch

        self.datasetSizes = {
            x: len(self.dataset[x]) for x in ['train', 'val']}

        # Define batch generator
        self.dataloaders = {}
        for x in ['train', 'val']:
            self.dataloaders[x] = torch.utils.data.DataLoader(self.dataset[x],
                                                              batch_size=self.num_examples_per_batch,
                                                              shuffle=True, num_workers=4)
        self.classNames = dataset['train'].classes
        self.numClasses = len(self.classNames)
        
        return self.dataloaders


    def define_model_resnet18(self, finetune=False):
        '''
            Define Resnet18 pretrained model.

            finetune: boolean
                If True, freezes convolutional layers to finetune network.
        '''
        self.finetune = finetune
        
        # Load pre-trained model
        self.model         = models.resnet18(pretrained=True)
        self.numFeatures   = self.model.fc.in_features

        # Freeze convolutional layers if finetuning
        if self.finetune:
            for param in self.model.parameters():
                param.requires_grad = False

        # Add FC layer
        # self.model.fc = nn.Linear(self.numFeatures, 2)
        # print("Num classes: ", self.numClasses)
        self.model.fc = nn.Linear(self.numFeatures, self.numClasses)
        # self.model.fc = nn.Softmax(dim=-1)
        self.model.to(self.device)

        return self.model


    def train(self, model, criterion, optimizer, scheduler, num_epochs=25):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.num_epochs = num_epochs

        self.bestModelWeights = copy.deepcopy(self.model.state_dict())

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
                    # print(self.inputs)
                    self.inputs = inputs.to(self.device)
                    self.labels = labels.to(self.device)
                    # print(self.inputs.size())
                    # print(self.labels.size())
                    # input()

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

                self.epochLoss = self.runningLoss / self.datasetSizes[phase]
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

        print("Best val accuracy: {:.4f}".format(self.bestAcc))

        # Load best model weights
        self.model.load_state_dict(self.bestModelWeights)
        return self.model

    def report_start(self):
        print
        