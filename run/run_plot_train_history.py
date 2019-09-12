import torch
import numpy                as np
import matplotlib.pyplot           as plt
from pathlib                import Path

import libs.utils           as utils
import libs.dirs            as dirs

# historyPath = Path(dirs.saved_models) / "full_dataset_history_no_finetune_25epoch.pickle"
historyPath = Path(dirs.saved_models) / "test_mnist_resnet18_history.pickle"

history = utils.load_pickle(historyPath)

print(history.keys())
valLoss = history['loss-val']
trainLoss = history['loss-train']
acc  = history['acc']
x = range(len(trainLoss))

plt.plot(x, valLoss, 'r.-', label="valLoss")
plt.plot(x, trainLoss, 'b.-', label="trainLoss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Training Loss history")
plt.show()