import os
import shutil
from tqdm           import tqdm
import numpy        as np
import pandas       as pd
from pathlib        import Path
from glob           import glob

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager
from libs.utils     import copy_files

class SampleImages:
    '''
        Sample images from a folder or an index determined by source.
        Sampled images are copied to destPath.
    '''
    def __init__(self, source, destPath):        
        self.source     = source
        self.destPath   = destPath
        self.percentage = None
        
        dirs.create_folder(self.destPath)


    def sample(self, percentage=0.01):
        self.percentage = percentage
        # If source is a directory, sample images, copy them to destPath
        # and save them in an index.
        if os.path.isdir(self.source):
            self._sample_from_folder()
        elif Path(self.source).suffix == "csv":
            self._sample_from_index()
        else:
            raise ValueError("Source must be a folder or csv index path.")


    def _sample_from_folder(self):
        def func_strip(x):         return Path(str(x).strip())
        # def func_relative_path(x): return x.relative_to(sourcePath)

        # Get video paths in dataset folder (all videos)
        self.imageList = glob(str(self.source) + "/**" + "/*.jpg", recursive=True)
        self.imageList = list(map(func_strip, self.imageList))
        # self.imageList = list(map(func_relative_path, self.imageList))

        print("")
        self.numImages = len(self.imageList)

        # Sample 1% of total images with normal distribution
        self.numSamples = int(self.numImages*0.01)
        self.sampleIndexes = np.random.choice(self.numImages, size=self.numSamples, replace=False)

        # Copy images to dest path
        print("Copying images...")
        for i in tqdm(range(self.numSamples)):
            # print("Copying image {}/{}".format(i+1, self.numSamples))
            index = self.sampleIndexes[i]
            image = self.imageList[index]
            newName = str(destPath / ("--".join(image.parts[3:-2]) + "--" + image.name))
            copy_files(image, newName)

        print("\nImage copying finished.")
        return self.numSamples
    

    def save_to_index(self):
        # TODO: IMPLEMENT
        raise NotImplementedError("Method not yet implemented.")


    def get_dest_folder_name(self):
        # TODO: IMPLEMENT
        raise NotImplementedError("Method not yet implemented.")


    def _sample_from_index(self):
        # TODO: IMPLEMENT
        raise NotImplementedError("Method not yet implemented.")


datasetName = "all_datasets_1s"
# locale      = dirs.febe_images  # Remote path
locale      = dirs.images       # Local path
sourcePath  = Path(locale + datasetName)
destPath    = Path(locale + "sampled_images_temp2/")

sampler = SampleImages(sourcePath, destPath)
sampler.sample()
