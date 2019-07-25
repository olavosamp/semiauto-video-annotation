import pytest
from pathlib                import Path
from glob                   import glob

import libs.dirs            as dirs
from libs.iteration_manager import SampleImages
from libs.utils             import copy_files

# class Test_maza(unittest.TestCase):
#     def test_maza1(self):
#         self.assertEqual(maza(4, 3), 6)

class Test_SampleImages():
    # def test_setup_SampleImages(self):
    #     self.testImagePath = Path(dirs.test_assets) / \
    #             "TVILL16-054_OK--DVD-1--Dive 420 16-02-24 19.32.32_C1.wmv FRAME 300.jpg"
        
    #     assert self.testImagePath.is_file()

    # def test_setup_get_image_dest_path(self):
    #     self.getImageDestPathTestFolder = Path(dirs.test) / "test_get_image_dest_path"
    #     dirs.create_folder(self.getImageDestPathTestFolder)
    #     self.testImagePath = Path(dirs.test_assets) / \
    #             "TVILL16-054_OK--DVD-1--Dive 420 16-02-24 19.32.32_C1.wmv FRAME 300.jpg"

    #     self.sourceFolder = (self.getImageDestPathTestFolder / "folder1") / "folder2"
    #     dirs.create_folder(self.sourceFolder)

    #     self.getImageDestPathTestImage = self.getImageDestPathTestFolder / self.testImagePath.name
        
    #     copy_files(self.testImagePath, self.getImageDestPathTestImage)

    #     assert self.getImageDestPathTestImage.is_file()
    #     assert self.getImageDestPathTestImage.suffix == ".jpg"


    def test_setup_SampleImages(self):
        sourceFolder = Path(dirs.test_assets) / "dataset_test"
        setupImageList = glob(str(sourceFolder) + "/**.jpg", recursive=True)

        assert len(setupImageList) == 2666


    def setup_sample_from_folder(self):
        self.sourceFolder = Path(dirs.test_assets) / "dataset_test"
        self.sampleImagesFolder = Path(dirs.test) / "test_sample_images"
        self.destFolderSFF = self.sampleImagesFolder / "test_sample_from_folder"
        dirs.create_folder(self.destFolderSFF)


    def teardown_sample_from_folder(self):
        sh.rmtree(self.destFolderSFF)


    def test_sample_from_folder(self):
        self.setup_sample_from_folder()
        assert self.destFolderSFF.is_dir()

        sampler = SampleImages(self.sourceFolder, self.destFolderSFF)

        # Verify image sampling and copying
        sampler.sample(percentage=0.01)
        imageList = glob(str(self.destFolderSFF) + "**.jpg", recursive=True)

        assert len(imageList) == 26

        self.outIndexPathSFF = self.destFolderSFF + "test_index_sample_from_file.csv"
        sampler.save_to_index(indexPath=self.outIndexPathSFF)

        self.outIndexSFF = pd.read_csv(self.outIndexPathSFF)
        assert self.outIndexPathSFF.is_file()
        assert self.outIndexSFF.shape[0] == 26
        
        self.teardown_sample_from_folder()
        # assert self.destFolderSFF.is_dir() == False
    