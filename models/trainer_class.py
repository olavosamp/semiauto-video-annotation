import time
import copy
import torch
import matplotlib
import torchvision
import random
import numpy                as np
import torch.nn             as nn
import sklearn.metrics      as skm
from tqdm                   import tqdm
from pathlib                import Path
from torchvision            import models
from torch.utils.data       import Sampler
from torchsummary           import summary

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
import models.utils         as mutils


class TrainModel:
    def __init__(self, model_path=None, seed=None, device_id=None, verbose=True):
        self.seed = seed
        if self.seed:
            mutils.set_torch_random_seeds(self.seed)

        self.model                  = None
        self.criterion              = None
        self.optimizer              = None
        self.num_epochs             = None
        self.finetune               = None
        self.num_examples_per_batch = None
        self.bestModelWeights       = None
        self.model_path             = model_path
        self.verbose                = verbose
        self.device_id              = device_id

        self.phases = ['train', 'val']

        # Select device
        self.set_device()


    def set_device(self):
        if torch.cuda.is_available():
            # Defaults to cuda:0
            deviceName = 'cuda:0'

            if self.device_id is not None:
                if isinstance(self.device_id, str) or isinstance(self.device_id, int):
                    # Set specified device number
                    deviceName = 'cuda:'+str(self.device_id)
        else:
            deviceName = 'cpu'

        if self.verbose:
            print("\nUsing device ", deviceName)
        self.device = torch.device(deviceName)


    def load_data(self, dataset, num_examples_per_batch=4,):
        '''
            Generate dataloaders for input dataset.

            dataset: dictionary of Dataset objects
                Dictionary of torch.utils.data.Dataset-derived objects. It must
                 contain the keys 'train' and 'val'.

            num_examples_per_batch: int
                Number of examples per batch.
        '''
        self.dataset = dataset
        self.num_examples_per_batch = num_examples_per_batch

        self.datasetSizes = {
            x: len(self.dataset[x]) for x in self.phases}

        # Define batch generator
        self.dataloaders = {}
        for x in self.phases:
            self.dataloaders[x] = torch.utils.data.DataLoader(self.dataset[x],
                                                              batch_size=self.num_examples_per_batch,
                                                              shuffle=True, num_workers=4)
        self.classNames = self.dataset['train'].classes
        self.numClasses = len(self.classNames)
        
        # Get class counts
        self.classIndexes = list(range(self.numClasses))
        self.classSizes = {}
        for phase in self.phases:
            self.classSizes[phase] = np.zeros(self.numClasses, dtype=int)
            targets = self.dataset[phase].targets
            for i in self.classIndexes:
                self.classSizes[phase][i] = np.sum(np.equal(targets, i))

        return self.dataloaders


    def define_model_resnet18(self, finetune=False, print_summary=False):
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
        # Use Linear layer for Cross Entropy loss
        self.model.fc = nn.Linear(self.numFeatures, self.numClasses)

        # Use Softmax layer with NLLoss
        # self.model.fc = nn.Softmax(dim=-1)

        # Use Sigmoid function with Binary Cross Entropy loss

        # Move model to device (must be done before constructing the optimizer)
        self.model.to(self.device)

        if print_summary:
            print("\n\t\tNetwork Summary")
            summary(self.model, (3, 224, 224), batch_size=128)
            print()
            # for idx, mod in enumerate(self.model.modules()):
            #     print(idx, "->", mod, " || ", mod.in_features())
            #     # print()
        
        # Load model weights, if provided
        if self.model_path is not None:
            self.model.load_state_dict(torch.load(self.model_path))

        return self.model


    def train(self, model, criterion, optimizer, scheduler=None, num_epochs=25):
        self.model            = model
        self.criterion        = criterion
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.num_epochs       = num_epochs
        self.bestAcc          = 0.
        self.bestLoss         = np.inf

        self.bestModelWeights = copy.deepcopy(self.model.state_dict())

        self.start      = time.time()
        
        # Training loop
        self.accHist         = {'train': [],
                                'val':   []}
        self.f1ScoreHist     = {'train': [],
                                'val':   []}
        self.lossHist        = {'train': [],
                                'val':   []}
        self.confMatHist = []
        for self.epoch in range(self.num_epochs):
            print("\n-----------------------")
            print("Epoch {}/{}".format(self.epoch+1, self.num_epochs))

            self.totalPreds      = {'train': [],
                                    'val':   []}
            self.totalLabels     = {'train': [],
                                    'val':   []}
            for phase in self.phases:
                trainPhase = (phase == 'train')
                if trainPhase:
                    self.model.train()   # Set model to training mode
                else:
                    self.model.eval()    # Set model to evaluate mode

                self.runningLoss     = 0.

                # Iterate over minibatches
                for inputs, labels in tqdm(self.dataloaders[phase]):
                    self.inputs = inputs.to(self.device)
                    self.labels = labels.to(self.device)
                    
                    # Reset gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(trainPhase):
                        # Forward propagation
                        self.outputs = self.model(self.inputs)
                        
                        # Get predictions as numerical class indexes
                        _, self.predictions = torch.max(self.outputs, 1)
                        self.loss = self.criterion(self.outputs, self.labels)

                        # Perform backpropagation only in train phase
                        if trainPhase:
                            self.loss.backward()
                            self.optimizer.step()
                            if self.scheduler:
                                self.scheduler.step()

                    # Get statistics
                    self.totalPreds[phase].extend(self.predictions.cpu().numpy())
                    self.totalLabels[phase].extend(self.labels.cpu().numpy())

                    self.runningLoss += self.loss.item() * self.inputs.size(0)

                # Get epoch metrics
                self.confMat   = skm.confusion_matrix(self.totalLabels[phase], self.totalPreds[phase])
                # self.epochAcc  = skm.accuracy_score(self.totalLabels[phase], self.totalPreds[phase])
                self.epochAcc = np.mean(mutils.compute_class_acc(self.confMat)) # Average of class accuracies
                self.epochLoss = self.runningLoss / self.datasetSizes[phase]
                _, _, self.epochF1, _ = skm.precision_recall_fscore_support(
                                            self.totalLabels[phase], self.totalPreds[phase])

                # Compute confusion matrix
                if phase == 'val':
                    self.confMatHist.append(self.confMat)

                self.f1ScoreHist[phase].append(self.epochF1)
                self.lossHist[phase].append(self.epochLoss)
                self.accHist[phase].append(self.epochAcc)

                if self.verbose:
                    print("{} Phase\n\
                Loss    : {:.4f}\n\
                Avg Acc : {:.4f}\n\
                F1      : {}".format(phase, self.epochLoss, self.epochAcc,
                                self.epochF1))

                # Save model if there is an improvement in evaluation metric
                if phase == 'val' and self.epochLoss < self.bestLoss:
                    self.bestLoss = self.epochLoss
                    self.bestAcc  = self.epochAcc
                    self.bestModelWeights = copy.deepcopy(self.model.state_dict())

        self.elapsedTime = time.time() - self.start
        print("\nTraining completed. Elapsed time: {:.0f}min{:.0f}".format(
                                            self.elapsedTime / 60, self.elapsedTime % 60))

        print("Best val loss:     {:.4f}".format(self.bestLoss))
        print("Best val accuracy: {:.4f} %".format(self.bestAcc*100))

        # Load best model weights
        self.model.load_state_dict(self.bestModelWeights)
        return self.model


    def save_history(self, histPath="./history_metrics.pickle"):
        self.histPath = histPath

        self.history = {
                    'loss-train':   self.lossHist['train'],
                    'loss-val':     self.lossHist['val'],
                    'f1-train':     self.f1ScoreHist['train'],
                    'f1-val':       self.f1ScoreHist['val'],
                    'acc-train':    self.accHist['train'],
                    'acc-val':      self.accHist['val'],
                    'conf-val':     self.confMatHist
        }

        utils.save_pickle(self.history, self.histPath)
        print("\nSaved train history to ", self.histPath)
        return self.history


    def model_inference(self, input_loader, save_path="inference_results.pickle"):
        print("Evaluating inputs...")
        print("Random state:\n", torch.random.get_rng_state())
        self.model.eval()

        self.outputs   = []
        self.imgHashes = []
        self.labelList = []
        for inputs, imgHash, labels in tqdm(input_loader):
            self.inputs = inputs.to(self.device)

            with torch.set_grad_enabled(False):
                self.batchOutput = self.model(self.inputs)
            
            # Store outputs in list
            self.outputs.extend(self.batchOutput.cpu().numpy())
            self.imgHashes.extend(imgHash)
            self.labelList.extend(labels)

        return self.outputs, self.imgHashes, self.labelList


    # def report_start(self):
    #     print

class MnistTrainer(TrainModel):
    def load_data(self, dataset, num_examples_per_batch=4):
        '''
            Generate dataloaders for input dataset.

            dataset: dictionary of Dataset objects
                Dictionary of torch.utils.data.Dataset-derived objects. It must
                 contain the keys 'train' and 'val'.

            num_examples_per_batch: int
                Number of examples per batch.
        '''
        self.dataset = dataset
        self.num_examples_per_batch = num_examples_per_batch

        self.datasetSizes = {
            x: len(self.dataset[x]) for x in self.phases}

        # Define batch generator
        self.dataloaders = {}
        for x in self.phases:
            self.dataloaders[x] = torch.utils.data.DataLoader(self.dataset[x],
                                                              batch_size=self.num_examples_per_batch,
                                                              shuffle=True, num_workers=4)
        self.classNames = self.dataset['train'].classes
        self.numClasses = len(self.classNames)
        
        # Get class counts
        self.classIndexes = list(range(self.numClasses))
        self.classSizes = {}
        for phase in self.phases:
            self.classSizes[phase] = np.zeros(self.numClasses, dtype=int)
            targets = self.dataset[phase].targets.numpy()
            for i in self.classIndexes:
                self.classSizes[phase][i] = np.sum(np.equal(targets, i))

        return self.dataloaders


# class IterLoopTrainer(TrainModel):
#     def load_data(self, dataset, split_percentages=[0.75, 0.15, 0.1], num_examples_per_batch=4):
#         '''
#             Generate dataloaders for input dataset.

#             dataset: dict-like dataset object
#                 Dataset dict-like object that must contain 'train' and 'val' keys.

#             split_percentages: list of positive floats
#                 List of dataset split percentages. Following the order [train, validation, test],
#                 each number represents the percentage of examples that will be allocated to
#                 the respective set.

#             num_examples_per_batch: int
#                 Number of examples per batch.
            
#             Returns:
#                 self.dataloader:
#                     Torch DataLoader constructed with input dataset and parameters.
#         '''
#         def SubsetSampler(Sampler):
#             def __init__(self, mask):
#                 self.mask = mask
#             def __iter__(self):
#                 return (self.indexes[i] for i in torch.nonzero(self.mask))
#             def __len__(self):
#                 return len(self.mask)

#         self.dataset = dataset
#         self.num_examples_per_batch = num_examples_per_batch

#         def data_split_indexes(data, split_percentages):
#             dataLen = len(data)

#             splitLen = [ceil(x) for x in split_percentages]

#             indexRef = np.random.shuffle(range(dataLen))
#             # TODO: Add support for split_percentages of arbitrary size
#             splitIndexes = []
#             valIndexes   = indexRef[:splitLen[1]]
#             trainIndexes = indexRef[splitLen[1]:]
#             splitIndexes.append(trainIndexes)
#             splitIndexes.append(valIndexes)
        

#         self.dataloaders = {}
#         self.dataloaders['train'] = SubsetSampler(self.trainIndexes)
#         self.dataloaders['val']   = SubsetSampler(self.valIndexes)
#         self.dataloaders['test']  = SubsetSampler(self.testIndexes)

#         # To sample shuffling the data, use
#         # torch.utils.data.DataLoader(trainset, batch_size=4, sampler=SubsetRandomSampler(
#         #     np.where(mask)[0]), shuffle=False, num_workers=2)

#         self.datasetSizes = {
#             x: len(self.dataset[x]) for x in ['train', 'val']}

#         # Define batch generator
#         self.dataloaders = {}
#         for x in ['train', 'val']:
#             self.dataloaders[x] = torch.utils.data.DataLoader(self.dataset[x],
#                                                                 batch_size=self.num_examples_per_batch,
#                                                                 shuffle=True, num_workers=4)
#         self.classNames = dataset['train'].classes
#         self.numClasses = len(self.classNames)
        
#         return self.dataloaders
