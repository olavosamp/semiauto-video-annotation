import time
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from glob                   import glob

import libs.dirs            as dirs
import libs.commons         as commons
import libs.utils           as utils


videoFolder = dirs.base_videos

videoList = utils.get_file_list(videoFolder, ext_list=commons.videoFormats)

start       = time.time()
hashTable   = utils.make_videos_hash_list(videoList, filepath_column='LocalisedVideoPath')
hashTime    = time.time() - start

print("Elapsed time to hash: {:.2f} seconds.".format(hashTime))
print(hashTable.shape)
print(hashTable.head())

destCsvPath = Path(dirs.hashtable)
hashTable.to_csv(destCsvPath, index=False)