import os
import shutil
import numpy        as np
import pandas       as pd
from pathlib        import Path
from glob           import glob

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager

def copy_files(source, destination):
    if os.path.isfile(source):
        shutil.copy2(source, destination)
        return True
    else:
        return False
    #     print("Source file not found:\n", source)

datasetName = "all_datasets_1s"
locale      = dirs.febe_images  # Remote path
# locale      = dirs.images       # Local path
imagePath   = Path(locale + datasetName)
destPath    = Path(locale + "sampled_images_temp/")

dirs.create_folder(destPath)

f = lambda x: Path(str(x).strip())
h = lambda x: x.relative_to(imagePath)

# Get video paths in dataset folder (all videos)
imageList = glob(str(imagePath) + "/**" + "/*.jpg", recursive=True)
imageList = list(map(f, imageList))
# imageList = list(map(h, imageList))

print("")
numImages = len(imageList)

# Sample 1% of total images with normal distribution
numSamples = int(numImages*0.01)
sampleIndexes = np.random.choice(numImages, size=numSamples, replace=False)

# Copy images to dest path
for i in range(numSamples):
    print("Copying image {}/{}".format(i+1, numSamples))
    index = sampleIndexes[i]
    image = imageList[index]
    newName = str(destPath / ("--".join(image.parts[3:-2]) + "--" + image.name))
    # print(image)
    # print(newName)
    # input()
    copy_files(image, newName)

print("\nImage copying finished.")
