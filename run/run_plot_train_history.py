import torch
import numpy                as np
import matplotlib.pyplot           as plt
from pathlib                import Path

import libs.utils           as utils
import libs.dirs            as dirs

historyPath = Path(dirs.saved_models) / "full_dataset_history_no_finetune_1000epochs.pickle"
# historyPath = Path(dirs.saved_models) / "test_mnist_resnet18_history_no_finetune.pickle"

resultsFolder = Path(dirs.results) / historyPath.stem
dirs.create_folder(resultsFolder)

history = utils.load_pickle(historyPath)

print(history.keys())
valLoss     = history['loss-val']
trainLoss   = history['loss-train']
trainAcc    = history['acc-train']
valAcc      = history['acc-val']
trainF1     = np.array((history['f1-train']))[:, 0]
valF1       = np.array((history['f1-val']))[:, 0]
# print(trainF1.shape)
# exit()
x = range(len(trainLoss))

# print(history['f1-train'])
# print(history['f1-val'])

fig = plt.figure(figsize=(24, 18))
plt.plot(x, valLoss, 'r.-', label="valLoss")
plt.plot(x, trainLoss, 'b.-', label="trainLoss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Training Loss history")
# plt.show()
fig.savefig(resultsFolder / "loss_history.pdf", orientation='portrait', bbox_inches='tight')

fig = plt.figure(figsize=(24, 18))
plt.plot(x, valAcc, 'r.-', label="valAcc")
plt.plot(x, trainAcc, 'b.-', label="trainAcc")
plt.legend()
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.title("Training Acc history")
# plt.show()
fig.savefig(resultsFolder / "accuracy_history.pdf", orientation='portrait', bbox_inches='tight')


fig = plt.figure(figsize=(24, 18))
plt.plot(x, valF1, 'r.-', label="valF1")
plt.plot(x, trainF1, 'b.-', label="trainF1")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Training F1 history, class 0")
# plt.show()
fig.savefig(resultsFolder / "f1_history.pdf", orientation='portrait', bbox_inches='tight')

# total 3918 imagens
# Treino 85%
#    Evento     1224
#    Nao Evento 2106
# Val    15%
#    Evento     224
#    Nao Evento 364
