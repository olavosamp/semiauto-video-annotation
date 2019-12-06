from libs.iteration_manager import IterationManager
from libs.index             import IndexManager

import libs.dirs            as dirs

iterFolder          = Path(dirs.iter_folder) / "test_loop/iteration_1/"
indexPath           = testFolder / "sampled_images.csv"
newLabelsPath       = testFolder / "sampled_images_labels.csv"

# Sample images

# Label images w/ interface
# Create sampled_images_labels.csv

# Add frame hash to labels file
# add_frame_hash_to_labels_file

# Merge interface labels file with index file
ind = IndexManager(indexPath)
newLabelsIndex = translate_interface_labels_file(newLabelsPath)
ind.merge_annotations(newLabelsIndex)

# Merge sampled_images index with existing labeled dataset
