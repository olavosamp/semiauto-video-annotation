from pathlib                import Path

import libs.dirs            as dirs
from libs.dataset_utils     import extract_dataset

if __name__ == "__main__":
    ## Dataset settings
    datasetName = "test_all_datasets_1s"

    # Local dataset
    datasetPath   = dirs.base_videos
    destFolder    = dirs.images+datasetName
    # Remote dataset
    # datasetPath   = dirs.febe_base_videos
    # destFolder    = dirs.febe_images+datasetName

    newIndexPath  = Path(dirs.root) / "index" / "test_index.csv"

    unlabeledIndex = extract_dataset(datasetPath, destFolder,
                                    datasetName="unlabeled_dataset_test",
                                    indexPath="auto")

    print("Index shape: ", unlabeledIndex.index.shape)
