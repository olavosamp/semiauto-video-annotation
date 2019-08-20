from pathlib                import Path

import libs.dirs            as dirs
from libs.iteration_manager import IterationManager

loopFolder = "../annotation_loop/test_loop/"
unlabeledIndexPath = Path(dirs.index) / "unlabeled_index_2019-8-16_11-37-59_HASHES.csv"
sourceFolder = Path(dirs.images) / "all_datasets_1s"

iterManager = IterationManager(sourceFolder, unlabeledIndexPath, loopFolder=loopFolder)
iterManager.new_iteration()
iterManager.sample_images()