import numpy                as np
from pathlib                import Path

import libs.utils           as utils
import libs.dirs            as dirs
from libs.vis_functions     import plot_model_history

iteration = int(input("Enter iteration number.\n"))
epochs    = int(input("Enter number of epochs.\n"))
rede        = 1

datasetName = "full_dataset_softmax"

savedModelsFolder    = Path(dirs.saved_models) / \
    "{}_rede_{}/iteration_{}".format(datasetName, rede, iteration)

historyPath = savedModelsFolder \
    / "history_{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(datasetName, epochs, rede, iteration)

resultsFolder        = Path(dirs.results) / historyPath.stem
nameEnd  = "history_{}_epochs_rede_{}_iteration_{}.pdf".format(epochs, rede, iteration)
lossName = "loss_"     + nameEnd
accName  = "accuracy_" + nameEnd
f1Name   = "f1_"       + nameEnd

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
