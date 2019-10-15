import torch
import numpy                as np
import torch.nn             as nn
import torch.optim          as optim
import torchvision.datasets as datasets
from pathlib                import Path
from torchvision            import transforms

import libs.dirs            as dirs
import libs.utils           as utils
from models.trainer_class   import TrainModel


if __name__ == "__main__":
    numImgBatch = 256
    numEpochs   = 25
    iteration   = 2

    modelPath = dirs.saved_models + \
            "full_dataset_no_finetune_{}_epochs_rede1_iteration_{}.pt".format(numEpochs, iteration)
    historyPath = dirs.saved_models + \
            "history_full_dataset_no_finetune_{}_epochs_rede1_iteration_{}.pickle".format(numEpochs, iteration)

    # Dataset root folder
    # Images should have paths following
    #   root/classA/img.jpg
    #   root/classB/img.jpg
    #   ...
    datasetPath = Path(dirs.iter_folder) / "full_dataset/iteration_{}/sampled_images/"

    # ImageNet statistics
    # No need to normalize pixel range from [0, 255] to [0, 1] because
    # ToTensor transform already does that
    mean    = [0.485, 0.456, 0.406]#/255
    std     = [0.229, 0.224, 0.225]#/255

    # Idea: compare with actual data statistics
    
    # Set transforms
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

    # Load Dataset objects for train and val sets from folder
    sets = ['train', 'val']
    imageDataset = {}
    for phase in sets:
        f = datasetPath / phase
        imageDataset[phase] = datasets.ImageFolder(str(f),
                                                   transform=dataTransforms[phase],
                                                   is_valid_file=utils.check_empty_file)

    datasetLen = len(imageDataset['train']) + len(imageDataset['val'])

    # Instantiate trainer object
    trainer = TrainModel()

    # Perform training
    trainer.load_data(imageDataset, num_examples_per_batch=numImgBatch)

    modelFineTune = trainer.define_model_resnet18(finetune=False)

    # Set optimizer and Loss criterion
    criterion = nn.CrossEntropyLoss()
    optimizerFineTune = optim.Adam(modelFineTune.parameters())

    # # Scheduler for learning rate decay
    # expLrScheduler = optim.lr_scheduler.StepLR(optimizerFineTune, step_size=7, gamma=0.1)

    modelFineTune = trainer.train(modelFineTune, criterion,
                                            optimizerFineTune, scheduler=None, num_epochs=numEpochs)
    history = trainer.save_history(historyPath)

    # Save model
    torch.save(modelFineTune.state_dict(), modelPath)
