import os
import numpy        as np
import pandas       as pd
from tqdm           import tqdm
from pathlib        import Path
from glob           import glob
from datetime       import datetime

import libs.dirs    as dirs
import libs.commons as commons
# from libs.index     import IndexManager
import libs.utils   as utils


class IterInfo:
    def __init__(self, datasetFolder, indexPath, iterFolder):
        self.datasetFolder      = datasetFolder
        self.iterFolder         = iterFolder
        self.indexPath          = indexPath
        self.iteration          = 0
        self.completed_iter     = False


class IterationManager:
    '''
        Iteration loop organizer class.

        Manages a iteration loop folder, containing a iter_info.pickle file that saves a IterInfo
        object that has details about the current loop state.

        Constructor arguments:
            unlabeledFolder:
                Folder containing the unlabeled images.
            unlabeledIndexPath:
                Path to the csv Index of the unlabeled images in unlabeledFolder
            loopFolder:
                Folder where the iteration folders and info will be saved

    '''
    def __init__(self, unlabeledFolder, unlabeledIndexPath, loopFolder=dirs.iter_folder):
        self.unlabeledFolder        = unlabeledFolder
        self.unlabeledIndexPath     = unlabeledIndexPath
        self.loopFolder             = Path(loopFolder)
        self.iterInfoPath           = self.loopFolder / "iter_info.pickle"

        self.load_info()


    def load_info(self):
        if self.iterInfoPath.is_file():
            self.iterInfo = utils.load_pickle(self.iterInfoPath)
        else:
            self.iterInfo = IterInfo(self.unlabeledFolder, self.unlabeledIndexPath, self.loopFolder)
            dirs.create_folder(self.loopFolder)

            utils.save_pickle(self.iterInfo, self.iterInfoPath)

        return self.iterInfo


    def new_iteration(self):
        '''
            create new iteration folder v
            sample new images           v
            update iter_info            v
        label images
        merge new labels (manual) to annotated dataset
        train model
        set boundaries
        automatic annotation
        merge new labels (automatic) to annotated dataset
            update iter_info, iteration complete

            Executes the following operations:
                Check if it is the first iteration;
                Load base index, create folders and iter_info;
                Sample images
        '''
        if self.iterInfo.completed_iter == False and self.iterInfo.iteration != 0:
            raise ValueError("Current iteration has not finished. Resolve it and try again.")

        self.iterInfo.iteration += 1
        self.iterInfo.completed_iter = False
        print("Starting iteration {}.".format(self.iterInfo.iteration))

        self.iterInfo.currentIterFolder = self.loopFolder / "iteration_{}".format(self.iterInfo.iteration)

        dirs.create_folder(self.iterInfo.currentIterFolder)
        print("Iteration setup finished.\nCall sample_images method for next step: sample and label images.")


    def sample_images(self, seed=None):
        '''
            Sample a percentage (1%) of the unlabeled images for labeling.

            Saves sampled images to 'iteration_#/sampled_images/'.
            Sampled images index is saved to 'iteration_#/sampled_images.csv'.
        '''
        self.samplesIndexPath = self.iterInfo.currentIterFolder / \
            "sampled_images_iteration_{}.csv".format(self.iterInfo.iteration)

        # Check if samples index already exists: probably means sample_images was
        # already executed this iteration
        if self.samplesIndexPath.is_file():
            raise FileExistsError(
                "Sampled index already exists.\nHas sampling been already performed this iteration?\n \
                To perform new sampling, delete sampled_images folder and index and run sample_images\
                 method again.")

        # TODO: REMEMBER to Remove fixed seed when using sampler outside of testing
        self.sampler = SampleImages(self.unlabeledIndexPath,
                                    self.iterInfo.currentIterFolder, seed=seed)

        self.sampler.sample(percentage=0.01)
        self.sampler.save_to_index(self.samplesIndexPath)


    def merge_labeled_dataset(self):
        pass

    def train_model(self):
        pass


class SampleImages:
    '''
        Sample images from a folder or an index determined by source.
        Sampled images are copied to destFolder / 'sampled_images'/.
    '''

    def __init__(self, source, destFolder, seed=None, verbose=True):
        self.date        = datetime.now()
        self.source      = Path(source)
        self.destFolder  = Path(destFolder)
        self.imageFolder = self.destFolder / "sampled_images"
        self.percentage  = None
        self.seed        = seed
        self.verbose     = verbose
        self.index       = None

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
        
        if self.verbose:
            print("{}/{} images copied to \"{}\".".format(self.numSuccess, self.numSamples, self.destFolder))
            if self.numSuccess != self.numSamples:
                print("{} image paths did not exist and were not copied.".format(self.numSamples-self.numSuccess))


    def _sample_routine(self):
        self.numImages = len(self.imageList)

        # Sample 1% of total images with normal distribution
        self.numSamples = int(self.numImages*self.percentage)
        self.sampleIndexes = np.random.choice(self.numImages, size=self.numSamples, replace=False)

        # Copy images to dest path
        print("Copying images...")
        self.imageSourcePaths = np.array(self.imageList)[self.sampleIndexes]
        self.imageDestPaths   = []
        self.numSuccess       = 0
        for i in tqdm(range(self.numSamples)):
            imagePath = self.imageSourcePaths[i]
            destPath = self.get_image_dest_path(imagePath)

            success = utils.copy_files(imagePath, destPath)
            self.numSuccess += success
            self.imageDestPaths.append(destPath)

        print("\nImage copying finished.")


    def _sample_from_folder(self):
        self.index = None

        # Get video paths in dataset folder (all videos)
        self.imageList = glob(str(self.source) + "/**" + "/*.jpg", recursive=True)
        self.imageList = list(map(utils.func_strip, self.imageList))

        self._sample_routine()
        return self.imageSourcePaths


    def _sample_from_index(self):
        self.index = pd.read_csv(self.source, dtype=str)
        self.imageList = self.index.loc[:, 'FramePath'].values
        self.imageList = list(map(utils.func_strip, self.imageList))

        self._sample_routine()
        return self.imageSourcePaths


    def save_to_index(self, indexPath='auto'):
        ''' 
            Saves sampled images information to csv index file.

            Optional Argument:
                indexPath: filepath. The index csv file is saved to indexPath. If indexPath is 'auto',
                it will save to destFolder / 'sample_index_' + <date string> + '.csv'. 
        '''
        if self.index is None:
            # Index does not exist: assemble new index
            self.index = pd.DataFrame({"FramePath": self.imageDestPaths,
                                       "OriginalPath": self.imageSourcePaths})
        else:
            # Index already exists and has been already initialized/loaded:
            # append new data to existing index
            self.index = self.index.loc[self.sampleIndexes, :]
        
        if indexPath is 'auto':
            # indexPath has not been passed: create a destination path
            self.indexPath = self.destFolder / ("sample_index_" + utils.get_time_string(self.date) + ".csv")
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
