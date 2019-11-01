import torch
import numpy                as np
import matplotlib.pyplot    as plt
from pathlib                import Path

import libs.utils           as utils
import libs.dirs            as dirs
from libs.vis_functions         import plot_model_history

iteration   = 3
epochs      = 500
rede        = 1

datasetName = "full_dataset_softmax"

savedModelsFolder    = Path(dirs.saved_models) / \
    "{}_rede_{}/iteration_{}".format(datasetName, rede, iteration)

historyPath = savedModelsFolder \
    / "history_{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(datasetName, epochs, rede, iteration)

resultsFolder        = Path(dirs.results) / historyPath.stem
lossName = "loss_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)
accName  = "accuracy_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)
f1Name   = "f1_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)

if not(historyPath.is_file()):
    print("History file does not exist.\nFile:\n", historyPath)
    print("\nExiting program.")
    exit()

dirs.create_folder(resultsFolder)

history = utils.load_pickle(historyPath)

print(history.keys())
valLoss     = history['loss-val']
trainLoss   = history['loss-train']
trainAcc    = history['acc-train']
valAcc      = history['acc-val']
trainF1     = np.array((history['f1-train']))[:, 0]
valF1       = np.array((history['f1-val']))[:, 0]

plot_model_history([trainLoss, valLoss], data_labels=["Train Loss", "Val Loss"], xlabel="Epochs",
                     ylabel="Loss", title="Training loss history", save_path=resultsFolder / lossName,
                     show=False)

plot_model_history([trainAcc, valAcc], data_labels=["Train Acc", "Val Acc"], xlabel="Epochs",
                     ylabel="Acc", title="Training accuracy history", save_path=resultsFolder / accName,
                     show=False)

plot_model_history([trainF1, valF1], data_labels=["Train F1", "Val F1"], xlabel="Epochs",
                     ylabel="F1", title="Training F1 history", save_path=resultsFolder / f1Name,
                     show=False)

print("\nSaved results to ", resultsFolder)
