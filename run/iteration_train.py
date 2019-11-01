from pathlib                    import Path

import libs.dirs                as dirs
import libs.commons             as commons
import models.utils             as mutils

if __name__ == "__main__":
    iteration = int(input("Enter iteration number."))
    seed           = 42
    # iteration      = 3
    rede           = 1
    epochs         = 500
    trainBatchSize = 256

    datasetName = "full_dataset_softmax"

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "{}/iteration_{}/".format(datasetName, iteration)

    iterFolder           = get_iter_folder(iteration)
    sampledImageFolder   = iterFolder / "sampled_images"
    sampledImageFolder   = iterFolder / "sampled_images"
    savedModelsFolder    = Path(dirs.saved_models) / \
        "{}_rede_{}/iteration_{}".format(datasetName, rede, iteration)

    modelPath            = savedModelsFolder / \
        "{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(datasetName, epochs, rede, iteration)
    historyPath          = savedModelsFolder / \
        "history_{}_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(datasetName, epochs, rede, iteration)

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

    # TODO: Plot train history here. Encapsulate scripts to functions and put here

