## Plot all confusion matrixes
import numpy                as np
from glob                   import glob
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
import models.utils         as mutils
import libs.commons         as commons
from libs.vis_functions     import plot_confusion_matrix

''' Prints all confusion matrixes. Uses history files found under folder
results/history_<dataset>_rede_<net_number>_val_<val_type>/'''

if __name__ == "__main__":
    # net_type = dutils.get_input_network_type(commons.network_types)
    # val_type = dutils.get_input_network_type(commons.val_types, message='validation set')
    # rede = int(input("\nEnter net number.\n"))
    numEpochs   = 25

    for net_type in commons.network_types.values():
        for val_type in commons.val_types.values():
            for rede in commons.net_target_column.keys():
                # Dataset root folder
                datasetPath = Path(dirs.dataset) / "{}_dataset_rede_{}_val_{}".format(net_type, rede, val_type)
                # datasetPath = Path(dirs.dataset) / "semiauto_dataset_v1_rede_{}".format(rede)
                datasetName = datasetPath.stem
                confMatPath = dirs.results+ "/confusion_matrix/" + "confusion_matrix_" + str(datasetName) + ".jpg"

                modelFolder = Path(dirs.saved_models) / \
                        "{}_{}_epochs".format(datasetName, numEpochs)
                historyFolder = Path(dirs.saved_models) / \
                        "history_{}_{}_epochs".format(datasetName, numEpochs)

                # Load history files
                globString = str(historyFolder / "history_run*pickle")
                histFiles  = glob(globString)

                bestValLoss = np.inf
                bestConfMat = None
                for hist in histFiles:
                    history = utils.load_pickle(hist)

                    # Get best epoch results
                    bestValIndex = np.argmin(history['loss-val'])
                    valLoss  = history['loss-val'][bestValIndex]
                    if valLoss < bestValLoss:
                        bestValLoss = valLoss
                        bestValAcc  = history['acc-val'][bestValIndex]
                        bestConfMat = history['conf-val'][bestValIndex]
                
                if bestConfMat is None:
                    print("No history file found in folder{}.".format(historyFolder))
                    continue
                print(bestConfMat)
                title = "Confusion Matrix "+str(datasetName)

                labels = commons.net_labels[rede]

                plot_confusion_matrix(bestConfMat, title=title, labels=labels, normalize=False, show=False,
                                        save_path=confMatPath)
