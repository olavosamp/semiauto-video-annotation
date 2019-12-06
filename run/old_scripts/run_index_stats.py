import pandas       as pd
from pathlib        import Path

import libs.dirs    as dirs
from libs.index     import IndexManager

indexPath = Path("./index/main_index.csv")
ind1  = IndexManager(path=indexPath)
# print(ind1.index.info())

print("\n")
print("Unique images:          ", ind1.index.shape[0])
print("Unique source videos:   ", ind1.index[['VideoName']].nunique().values[0])
print("Unique source datasets: ", ind1.index[['OriginalDataset']].nunique().values[0])

uniqueTags = ind1.get_unique_tags()
tagCount = {}
for targetTag in uniqueTags:
    ind1  = IndexManager(path=indexPath)
    destFolder = Path(dirs.images) / 'temp' / targetTag

    # print("Selecting images of class: ", targetTag)
    selectDf = ind1.index.copy()
    f = lambda x: x.split('-')
    selectDf['TagList'] = ind1.index['Tags'].apply(f)

    selectIndex = [i for i,x in enumerate(selectDf['TagList']) if targetTag in x]
    ind1.index = selectDf.loc[selectIndex, :]
    # print(len(selectIndex), " images found.")
    tagCount[targetTag] = len(selectIndex)

print("\nTags:\n")
for tag, count in tagCount.items():
    print(tag.ljust(10), ":", "{}".format(count).rjust(5))
