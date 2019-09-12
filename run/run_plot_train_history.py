import torch
import numpy                as np
import matplotlib           as plt
from pathlib                import Path

import libs.utils           as utils
import libs.dirs            as dirs


historyPath = Path(dirs.saved_models) / "full_dataset_history_no_finetune.pickle"
history = utils.load_pickle(historyPath)

print(history.keys())
valLoss = history['loss-val']
trainLoss = history['loss-train']
acc  = history['acc']
# x = range(len(trainLoss))

print(valLoss)
print(trainLoss)
print(acc)
# plt.plot(x, valLoss, 'r.')
# plt.plot(x, trainLoss, 'b.')
# plt.show()