from pathlib                    import Path
import numpy                    as np

import libs.dirs                as dirs
import libs.commons             as commons
import models.utils             as mutils
import libs.utils               as utils
import libs.dataset_utils       as dutils
from libs.vis_functions         import plot_model_history

if __name__ == "__main__":
    iteration = int(input("Enter iteration number.\n"))
    seed           = np.random.randint(0, 100)
    rede           = 2
    epochs         = 150
    trainBatchSize = 256

    datasetName = "full_dataset_rede_{}".format(rede)

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "{}/iteration_{}/".format(datasetName, iteration)

    iterFolder           = get_iter_folder(iteration)
    sampledImageFolder   = iterFolder / "sampled_images"
    savedModelsFolder    = Path(dirs.saved_models) / \
        "{}/iteration_{}".format(datasetName, iteration)

    modelPath            = savedModelsFolder / \
        "{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(datasetName, epochs, rede, iteration)
    historyPath          = savedModelsFolder / \
        "history_{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(datasetName, epochs, rede, iteration)

    historyFolder = Path(dirs.results) / historyPath.stem
    lossPath = historyFolder / \
                "loss_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)
    accPath  = historyFolder / \
                "accuracy_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)
    f1Path   = historyFolder / \
                "f1_history_{}_epochs_rede_{}_iteration{}.pdf".format(epochs, rede, iteration)
    seedLogPath           = iterFolder / "seeds.txt"
    
    dirs.create_folder(historyFolder)

    ## Train model
    print("\nSTEP: Train model.")
    # ImageNet statistics
    mean    = commons.IMAGENET_MEAN
    std     = commons.IMAGENET_STD 

    # Set transforms
    dataTransforms = mutils.resnet_transforms(mean, std)
    history, modelFineTune = mutils.train_network(sampledImageFolder, dataTransforms, epochs=epochs,
                                            batch_size=trainBatchSize,
                                            model_path=modelPath,
                                            history_path=historyPath,
                                            seed=seed)

    print("\nPlot train history.")
    history = utils.load_pickle(historyPath)

    valLoss     = history['loss-val']
    trainLoss   = history['loss-train']
    trainAcc    = history['acc-train']
    valAcc      = history['acc-val']
    trainF1     = np.array((history['f1-train']))[:, 0]
    valF1       = np.array((history['f1-val']))[:, 0]

    plot_model_history([trainLoss, valLoss], data_labels=["Train Loss", "Val Loss"], xlabel="Epochs",
                        ylabel="Loss", title="Training loss history", save_path=lossPath,
                        show=False)

    plot_model_history([trainAcc, valAcc], data_labels=["Train Acc", "Val Acc"], xlabel="Epochs",
                        ylabel="Acc", title="Training accuracy history", save_path=accPath,
                        show=False)

    plot_model_history([trainF1, valF1], data_labels=["Train F1", "Val F1"], xlabel="Epochs",
                        ylabel="F1", title="Training F1 history", save_path=f1Path,
                        show=False)

    print("\nSaved results to folder ", historyFolder)
    
    # Save sample seed
    dutils.save_seed_log(seedLogPath, seed, "train")

