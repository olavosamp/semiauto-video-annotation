import time
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from glob                   import glob

import libs.dirs            as dirs
import libs.commons         as commons
from libs.utils             import (file_hash,
                                    replace_backslashes,
                                    make_video_hash_list
                                    )


videoFolder = dirs.base_videos

start = time.time()
hashTable = make_video_hash_list(videoFolder)
hashTime = time.time() - start

print("Elapsed time to hash: {:.2f} seconds.".format(hashTime))
print(hashTable.shape)
print(hashTable.head())

destCsvPath = Path("./index/localised_video_path_listTEST.csv")
hashTable.to_csv(destCsvPath, index=False)