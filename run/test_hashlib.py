import hashlib
import numpy                as np
import pandas               as pd
from tqdm                   import tqdm
from pathlib                import Path
from glob                   import glob

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import IndexManager
from libs.utils             import file_hash


indexPath  = Path(dirs.root) / "index" / "test_index.csv"

ind2  = IndexManager(path=indexPath)

print(ind2.indexExists)

hashedIndex = ind2.index.copy()
# hashedIndex['VideoHash'] = hashedIndex['VideoPath'].map(file_hash)
numVideos = ind2.get_index_len()

videoList = ind2.get_video_path_list()

numVideos = len(videoList)
hashList = []
for i in tqdm(range(numVideos)):
    hashList.append(file_hash(videoList[i]))
print("hash1: ", np.shape(hashList)[0])
print(hashList)

hashList2 = ind2.get_video_hash_list()
print("hash2: ", np.shape(hashList2)[0])
print(hashList2)

diff = set(hashList) - set(list(hashList2['VideoHash']))
print(len(diff))
print(diff)

