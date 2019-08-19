import os
import numpy        as np
import pandas       as pd
from tqdm           import tqdm
from pathlib        import Path
from glob           import glob
from datetime       import datetime

import libs.dirs    as dirs
import libs.commons as commons
from libs.index     import IndexManager
from libs.utils     import *


def func_strip(x):         return Path(str(x).strip())
def func_relative_path(x): return x.relative_to(sourcePath)


class IterInfo:
    def __init__(self, baseDataset, iter_folder):
        self.baseDataset        = baseDataset
        self.iter_folder        = iter_folder
        self.iteration          = 0
        self.completed_iter     = False


class IterationManager:
    def __init__(self, unlabeledFolder, iterFolder=dirs.iter_folder):
        self.unlabeledFolder = unlabeledFolder
        self.iterFolder      = Path(iterFolder)
        self.iterInfoPath    = self.iterFolder / "iter_info.txt"

        self.load_info()

    def load_info(self):
        if self.iterInfoPath.is_file():
            self.iterInfo = load_pickle(self.iterInfoPath)
        else:
            self.iterInfo = IterInfo(self.unlabeledFolder, self.iter_folder)
            create_folder(self.iterFolder)

            save_pickle(self.iterInfo, self.iterInfoPath)

        return self.iterInfo


    def new_iteration(self):
        '''
            Executes the following operations
        '''
        print("Finished iteration {}. Starting iteration {}.".format(
            self.iterInfo.iteration, self.iterInfo.iteration+1))
        self.iterInfo.iteration += 1
        self.iterInfo.completed_iter = False

        newFolder = self.iterFolder / "iteration_"+self.iterInfo.iteration
        create_folder(newFolder)




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
        self.imageList = self.index.loc[:, 'VideoPath'].values
        self.imageList = list(map(func_strip, self.imageList))

        self._sample_routine()
        return self.imageSourcePaths


    def save_to_index(self, indexPath=None):
        ''' 
            Saves sampled images information to csv index file.

            Optional Argument:
                indexPath: filepath. The index csv file is saved to indexPath. If indexPath is None, 
        '''
        if self.index is None:
            # Index does not exist: assemble new index
            self.index = pd.DataFrame({"FramePath": self.imageDestPaths,
                                       "OriginalPath": self.imageSourcePaths})
        else:
            # Index already exists and has been already initialized/loaded:
            # append new data to existing index
            self.index = self.index.loc[self.sampleIndexes, :]
        
        if indexPath is None:
            # indexPath has not been passed: create a destination path
            self.indexPath = self.destFolder / ("sample_index_" + get_time_string(self.date) + ".csv")
        else:
            # indexPath has been passed: use it as destination path
            self.indexPath = indexPath
            assert (Path(self.indexPath).suffix == ".csv"), "IndexPath must point to a csv file."

        self.index.to_csv(self.indexPath, index=False)
        return self.indexPath


    def get_image_dest_path(self, imagePath):
        '''
            Assemble destination path for a source image path.
            Destination path will use the pre-determined image folder and the source image's name.
        '''
        imagePath = Path(imagePath)
        destPath = self.imageFolder / imagePath.name
        return destPath
