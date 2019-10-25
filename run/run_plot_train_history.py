import torch
import numpy                as np
import matplotlib.pyplot    as plt
from pathlib                import Path

import libs.utils           as utils
import libs.dirs            as dirs

iteration   = 2
epochs      = 100
rede        = 1

historyPath = Path(dirs.saved_models) \
    / "history_full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(epochs, rede, iteration)
# historyPath = Path(dirs.saved_models) / "test_mnist_resnet18_history_no_finetune.pickle"

if not(historyPath.is_file()):
    print("History file does not exist.\nFile:\n", historyPath)
    print("\nExiting program.")
    exit()

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
lossName = "loss_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)
accName  = "accuracy_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)
f1Name   = "f1_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)

fig = plt.figure(figsize=(24, 18))
plt.plot(x, valLoss, 'r.-', label="Val Loss")
plt.plot(x, trainLoss, 'b.-', label="Train Loss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Training Loss history")
# plt.show()
fig.savefig(resultsFolder / lossName, orientation='portrait', bbox_inches='tight')

fig = plt.figure(figsize=(24, 18))
plt.plot(x, valAcc, 'r.-', label="Val Acc")
plt.plot(x, trainAcc, 'b.-', label="Train Acc")
plt.legend()
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.title("Training Accuracy history")
# plt.show()
fig.savefig(resultsFolder / accName, orientation='portrait', bbox_inches='tight')


fig = plt.figure(figsize=(24, 18))
plt.plot(x, valF1, 'r.-', label="Val F1")
plt.plot(x, trainF1, 'b.-', label="Train F1")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Training F1 history")
# plt.show()
fig.savefig(resultsFolder / f1Name, orientation='portrait', bbox_inches='tight')

print("\nSaved results to ", resultsFolder)
# total 3918 imagens
# Treino 85%
#    Evento     1224
#    Nao Evento 2106
# Val    15%
#    Evento     224
#    Nao Evento 364
