import os
import shutil
import numpy        as np
import pandas       as pd
from tqdm           import tqdm
from pathlib        import Path
from glob           import glob
from datetime       import datetime

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager
from libs.utils     import copy_files, get_time_string


def func_strip(x):         return Path(str(x).strip())
# def func_relative_path(x): return x.relative_to(sourcePath)


class SampleImages:
    '''
        Sample images from a folder or an index determined by source.
        Sampled images are copied to destFolder.
    '''

    def __init__(self, source, destFolder, seed=None):
        self.date        = datetime.now()
        self.source      = Path(source)
        self.destFolder  = Path(destFolder)
        self.imageFolder = self.destFolder / "images"
        self.percentage  = None
        self.seed        = seed

        np.random.seed(self.seed)
        dirs.create_folder(self.destFolder)
        dirs.create_folder(self.imageFolder)


    def sample(self, percentage=0.01):
        self.percentage = percentage
        if self.source.is_dir():
            # Sample files from directory
            self._sample_from_folder()

        elif self.source.suffix == ".csv":
            # Sample files from entries in a csv index
            self._sample_from_index()
        else:
            raise ValueError("Source must be a folder or csv index path.")


    def _sample_routine(self):
        self.numImages = len(self.imageList)

        # Sample 1% of total images with normal distribution
        self.numSamples = int(self.numImages*self.percentage)
        self.sampleIndexes = np.random.choice(self.numImages, size=self.numSamples, replace=False)

        # Copy images to dest path
        print("Copying images...")
        self.imageSourcePaths = np.array(self.imageList)[self.sampleIndexes]
        self.imageDestPaths   = []
        for i in tqdm(range(self.numSamples)):
            # print("Copying image {}/{}".format(i+1, self.numSamples))
            imagePath = self.imageSourcePaths[i]
            destPath = self.get_image_dest_path(imagePath)

            copy_files(imagePath, destPath)
            self.imageDestPaths.append(destPath)

        print("\nImage copying finished.")


    def _sample_from_folder(self):
        self.index = None

        # Get video paths in dataset folder (all videos)
        self.imageList = glob(str(self.source) + "/**" + "/*.jpg", recursive=True)
        self.imageList = list(map(func_strip, self.imageList))
        # self.imageList = list(map(func_relative_path, self.imageList))

        self._sample_routine()
        return self.imageSourcePaths
    

    def _sample_from_index(self):
        self.index = pd.read_csv(self.source, dtype=str)
        print(self.index.head())
        print(self.index.info())
        self.imageList = self.index.loc[:, 'VideoPath'].values
        self.imageList = list(map(func_strip, self.imageList))

        self._sample_routine()
        return self.imageSourcePaths


    def save_to_index(self, indexPath=None):
        ''' 
            Saves sampled images information to csv index file.
        '''
        if self.index is None:
            self.index = pd.DataFrame({"FramePath": self.imageDestPaths,
                                       "OriginalPath": self.imageSourcePaths})
        else:
            self.index = self.index.loc[self.sampleIndexes, :]
        
        if indexPath is None:
            self.indexPath = self.destFolder / ("sample_index_" + get_time_string(self.date) + ".csv")
        else:
            self.indexPath = indexPath
            assert (Path(self.indexPath).suffix == ".csv"), "IndexPath must point to a csv file."

        self.index.to_csv(self.indexPath, index=False)
        return self.indexPath


    # def get_image_dest_path(self, imagePath):
    #     tmp = imagePath.relative_to(self.source).parts
    #     destPath = self.imageFolder / "--".join(tmp)
    #     return destPath

    def get_image_dest_path(self, imagePath):
        imagePath = Path(imagePath)
        destPath = self.imageFolder / imagePath.name
        return destPath
