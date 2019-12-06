from pathlib                import Path

import libs.dirs            as dirs
from libs.dataset_utils     import extract_dataset

if __name__ == "__main__":
    ## Dataset settings
    datasetName = "unlabeled_dataset_local"
    newIndexPath  = Path(dirs.dataset) / datasetName / (datasetName+".csv")
    # newIndexPath  = Path(dirs.root) / "index" / "test_index.csv"

    # Local dataset
    datasetPath   = dirs.base_videos
    destFolder    = Path(dirs.dataset) / datasetName / "images"
    # Remote dataset
    # datasetPath   = dirs.febe_base_videos
    # destFolder    = dirs.febe_images+datasetName

    unlabeledIndex = extract_dataset(datasetPath, destFolder,
                                    datasetName=datasetName,
                                    indexPath=newIndexPath)

    print("Index shape: ", unlabeledIndex.index.shape)
