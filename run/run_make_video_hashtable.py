import time
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from glob                   import glob

import libs.dirs            as dirs
import libs.commons         as commons
from libs.utils             import *


videoFolder = dirs.base_videos

videoList = get_file_list(videoFolder, ext_list=commons.videoFormats)

start       = time.time()
hashTable   = make_video_hash_list(videoList, columnName='LocalisedVideoPath')
hashTime    = time.time() - start

print("Elapsed time to hash: {:.2f} seconds.".format(hashTime))
print(hashTable.shape)
print(hashTable.head())

destCsvPath = Path(dirs.hashtable)
hashTable.to_csv(destCsvPath, index=False)