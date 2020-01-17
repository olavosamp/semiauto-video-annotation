import numpy                as np
from glob                   import glob
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
import models.utils         as mutils
import libs.commons         as commons
from libs.vis_functions     import plot_confusion_matrix


# # Get best epoch results
# bestValIndex = np.argmin(history['loss-val'])
# bestValLoss  = history['loss-val'][bestValIndex]
# bestValAcc   = history['acc-val'][bestValIndex]
# confMat      = history['conf-val'][bestValIndex]

# TODO:Create function to get user input on reference or semiauto dataset selection

if __name__ == "__main__":
    rede = int(input("\nEnter net number.\n"))
    numEpochs   = 25

    # Dataset root folder
#     datasetPath = Path(dirs.dataset) / "reference_dataset_rede_{}".format(rede)
    datasetPath = Path(dirs.dataset) / "semiauto_dataset_v1_rede_{}".format(rede)
    datasetName = datasetPath.stem

    modelFolder = Path(dirs.saved_models) / \
            "{}_{}_epochs".format(datasetName, numEpochs)
    historyFolder = Path(dirs.saved_models) / \
            "history_{}_{}_epochs".format(datasetName, numEpochs)

    globString = str(historyFolder)+"history_run*pickle"
    histFiles  = glob(globString)

    # history = utils.load_pickle(historyFolder)
    # historyPath = historyFolder / "history_run_{}.pickle".format(i)