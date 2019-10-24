from pathlib                    import Path

import libs.dirs                as dirs
import libs.commons             as commons
import models.utils             as mutils

if __name__ == "__main__":
    seed           = 42
    iteration      = 2
    rede           = 1
    epochs         = 1000
    trainBatchSize = 256

    def get_iter_folder(iteration):
        return Path(dirs.iter_folder) / "full_dataset_softmax/iteration_{}/".format(iteration)

    iterFolder           = get_iter_folder(iteration)
    sampledImageFolder   = iterFolder / "sampled_images"
    savedModelsFolder    = Path(dirs.saved_models) / "full_dataset_rede_{}/iteration_{}".format(rede, iteration)

    modelPath            = savedModelsFolder / \
        "full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pt".format(epochs, rede, iteration)
    historyPath          = savedModelsFolder / \
        "history_full_dataset_no_finetune_{}_epochs_rede_{}_iteration_{}.pickle".format(epochs, rede, iteration)

    ## Train model
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

