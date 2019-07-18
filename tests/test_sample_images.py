import unittest
from pathlib                import Path

import libs.dirs            as dirs
from libs.iteration_manager import SampleImages
from libs.utils             import copy_files

# class Test_maza(unittest.TestCase):
#     def test_maza1(self):
#         self.assertEqual(maza(4, 3), 6)

class Test_SampleImages(unittest.TestCase):
    def setUp(self):
        self.testImagePath = Path(dirs.test_assets) / \
                "TVILL16-054_OK--DVD-1--Dive 420 16-02-24 19.32.32_C1.wmv FRAME 300.jpg"
        
        self.assertTrue( self.testImagePath.is_file())

    def setup_get_image_dest_path(self):
        self.getImageDestPathTestFolder = Path(dirs.test) / "test_get_image_dest_path"
        dirs.create_folder(self.getImageDestPathTestFolder)
        # self.testImagePath = Path(dirs.test_assets) / \
        #         "TVILL16-054_OK--DVD-1--Dive 420 16-02-24 19.32.32_C1.wmv FRAME 300.jpg"

        self.sourceFolder = (self.getImageDestPathTestFolder / "folder1") / "folder2"
        dirs.create_folder(self.sourceFolder)
        self.getImageDestPathTestImage = self.getImageDestPathTestFolder / self.testImagePath.name
        copy_files(self.testImagePath, self.getImageDestPathTestImage)

        self.assertTrue(self.getImageDestPathTestImage.is_file() and self.getImageDestPathTestImage.suffix == "jpg")

    # def test_get_image_dest_path(self):
        