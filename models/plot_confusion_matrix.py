# Plot all confusion matrixes
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

import libs.dirs as dirs
import libs.utils as utils
import libs.dataset_utils as dutils
import models.utils as mutils
import libs.commons as commons
from libs.vis_functions import plot_confusion_matrix

''' Prints all confusion matrixes. Uses history files found under folder
results/history_<dataset>_rede_<net_number>_val_<val_type>/'''

if __name__ == "__main__":
    numEpochs = 25
    normalize = True
    # net_type = dutils.get_input_network_type(commons.network_types)
    # val_type = dutils.get_input_network_type(commons.val_types, message='validation set')
    # rede = int(input("\nEnter net number.\n"))
    confMatFolder = Path(dirs.results) / "confusion_matrix"
    dfPath = confMatFolder / "stats_table"

    rede3List = []
    otherList = []
    for net_type in commons.network_types.values():
        for val_type in commons.val_types.values():
            for rede in commons.net_target_column.keys():
                # Dataset root folder
                datasetPath = Path(dirs.dataset) / \
                    "{}_dataset_rede_{}_val_{}".format(net_type, rede, val_type)
                datasetName = datasetPath.stem
                confMatPath = confMatFolder / \
                    ("confusion_matrix_" + str(datasetName) + ".pdf")

                modelFolder = Path(dirs.saved_models) / \
                    "{}_{}_epochs".format(datasetName, numEpochs)
                historyFolder = Path(dirs.saved_models) / \
                    "history_{}_{}_epochs".format(datasetName, numEpochs)

                # Load history files
                globString = str(historyFolder / "history_run*pickle")
                histFiles = glob(globString)

                bestValLoss = np.inf
                bestConfMat = None
                for hist in histFiles:
                    history = utils.load_pickle(hist)

                    # Get best epoch results
                    bestValIndex = np.argmin(history['loss-val'])
                    valLoss = history['loss-val'][bestValIndex]
                    if valLoss < bestValLoss:
                        bestValLoss = valLoss
                        bestValAcc = history['acc-val'][bestValIndex]
                        bestConfMat = history['conf-val'][bestValIndex]
                        bestValF1 = history['f1-val'][bestValIndex]

                if bestConfMat is None:
                    print("No history file found in folder{}.".format(historyFolder))
                    continue
                if val_type == 'ref':
                    valName = 'Reference'
                elif val_type == 'semiauto':
                    valName = 'Semiauto'
                netName = net_type[0].upper() + net_type[1:]
                title = "Level {} | Dataset {} | Val {}".format(rede, netName, valName)

                acc = mutils.compute_class_acc(bestConfMat)

                print("\n", title)
                print(bestConfMat)
                print("Mean Acc: ", np.mean(acc))
                print("Std  Acc: ", np.std(acc))
                print("Acc: ", acc)
                print("F1:  ", bestValF1)
                labels = commons.net_labels[rede]

                if rede == 3:
                    entrySeries = pd.Series({'rede': rede,
                                             'dataset': net_type,
                                             'validation': val_type,
                                             'mean_acc': np.mean(acc),
                                             'std_acc':  np.std(acc),
                                             'f1_anode':  bestValF1[0],
                                             'f1_damage': bestValF1[1],
                                             'f1_buried': bestValF1[2],
                                             'f1_flange': bestValF1[3],
                                             'f1_repair': bestValF1[4]
                                             })
                    rede3List.append(entrySeries)
                else:
                    entrySeries = pd.Series({'rede': rede,
                                             'dataset': net_type,
                                             'validation': val_type,
                                             'mean_acc': np.mean(acc),
                                             'std_acc':  np.std(acc),
                                             'f1_positive':  bestValF1[0],
                                             'f1_negative': bestValF1[1],
                                             })
                    otherList.append(entrySeries)

                plot_confusion_matrix(bestConfMat, title=title, labels=labels, normalize=normalize, show=False,
                                      save_path=confMatPath)
    statsDf = pd.DataFrame(rede3List)
    statsDf.to_excel(str(dfPath)+"_rede_3.xlsx")

    statsDf = pd.DataFrame(otherList)
    statsDf.to_excel(str(dfPath)+"_outros.xlsx")

