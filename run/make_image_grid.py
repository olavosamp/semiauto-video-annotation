import numpy                as np
# from glob                   import glob
from pathlib                import Path

import libs.dirs            as dirs
import libs.utils           as utils
# import libs.dataset_utils   as dutils
# import models.utils         as mutils
import libs.commons         as commons
# from libs.vis_functions     import plot_confusion_matrix

# targetFolder  = Path(dirs.images) / "image_grid/duct_positives"
# targetFolder  = (Path(dirs.images) / "sampled_images_sorted") / "not_duct"
targetFolder  = (Path(dirs.images) / "full_dataset_results_samples__95_ratio") / "NaoEvento"
imageGridPath = targetFolder.parent.with_name(targetFolder.stem+"_grid.pdf")

utils.image_grid(targetFolder, imageGridPath, upperCrop=55, lowerCrop=50, size_limit=9)

