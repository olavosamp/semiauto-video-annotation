import pytest
import shutil               as sh
import pandas               as pd
from pathlib                import Path
from glob                   import glob

import libs.dirs            as dirs
from libs.iteration_manager import SampleImages
from libs.utils             import copy_files, replace_backslashes


class Test_SampleImages():
    def test_setup_SampleImages(self):
        # Metatest if test assets are in place
        sourceFolder = Path(dirs.test_assets) / "dataset_test"
        setupImageList = glob(str(sourceFolder) + "/**.jpg", recursive=True)

        assert len(setupImageList) == 2666


    def setup_sample_from_folder(self):
        self.sourceFolder = Path(dirs.test_assets) / "dataset_test"
        self.sampleImagesFolder = Path(dirs.test) / "test_sample_images"
        self.destFolderSFF = self.sampleImagesFolder / "test_sample_from_folder"

        # Guarantee that the destination folder was created for this test only
        if self.destFolderSFF.is_dir():
            self.teardown_sample_from_folder()
        dirs.create_folder(self.destFolderSFF)


    def teardown_sample_from_folder(self):
        sh.rmtree(self.destFolderSFF)


    def test_sample_from_folder(self):
        self.setup_sample_from_folder()
        assert self.destFolderSFF.is_dir()

        self.sampler = SampleImages(self.sourceFolder, self.destFolderSFF)

        # Test image sampling and copying
        self.sampler.sample(percentage=0.01)

        globString = str(self.sampler.imageFolder) + "/**.jpg"
        globString = replace_backslashes(globString)
        imageList = glob(globString, recursive=True)

        assert len(imageList) == 26

        # Test saving samples to index
        self.outIndexPathSFF = self.sampler.imageFolder / "test_index_sample_from_file.csv"
        print("Saving index to\n", self.outIndexPathSFF)

        self.sampler.save_to_index(indexPath=self.outIndexPathSFF)

        self.outIndexSFF = pd.read_csv(self.outIndexPathSFF)
        assert self.outIndexPathSFF.is_file()
        assert self.outIndexSFF.shape[0] == 26
        
        self.teardown_sample_from_folder()