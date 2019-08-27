import pytest
import shutil               as sh
import pandas               as pd
from pathlib                import Path

import libs.dirs            as dirs
import libs.commons         as commons
from libs.index             import IndexManager
from libs.dataset_utils     import *
from libs.utils             import *


class Test_merge_annotations:
    def test_setup_merge_annotations(self):
        '''
            Check if test assets are in place and move files to active test folder.
        '''
        self.assetsFolder        = Path(dirs.test_assets) / "test_loop/iteration_0/"

        self.testFolder          = Path(dirs.test) / "test_loop/iteration_0/"
        self.indexPath           = self.testFolder / "sampled_images.csv"
        self.newLabelsPath       = self.testFolder / "sampled_images_labels.csv"

        fileList = get_file_list(str(self.assetsFolder))
        for f in fileList:
            fPath = Path(f)
            newPath = self.testFolder / fPath.relative_to(self.assetsFolder)
            dirs.create_folder(newPath.parent)
            copy_files(str(f), str(newPath))

        assert self.indexPath.is_file()
        assert self.newLabelsPath.is_file()


    def test_add_frame_hash(self):
        ''' Test if function adds FrameHash columns with correct hash. '''
        self.testFolder     = Path(dirs.test) / "test_loop/iteration_0/"
        self.newLabelsPath  = self.testFolder / "sampled_images_labels.csv"

        add_frame_hash_to_labels_file(self.newLabelsPath, framePathColumn='imagem')
        labelsDf = pd.read_csv(self.newLabelsPath)

        assert labelsDf.loc[13, 'FrameHash'] == '1bf13ea0c0537481aa1d1ed46b207783'


    def test_translate_interface_labels(self):
        self.testFolder          = Path(dirs.test) / "test_loop/iteration_0/"
        self.indexPath           = self.testFolder / "sampled_images.csv"
        self.newLabelsPath       = self.testFolder / "sampled_images_labels.csv"

        newLabelsIndex = translate_interface_labels_file(self.newLabelsPath)
        assert newLabelsIndex.loc[21, 'Tags'] == ["Duto", "Evento", "Flange"]


    def test_merge_annotations(self):
        #TODO: Finish this test when there are example labeled index csvs.
        self.testFolder          = Path(dirs.test) / "test_loop/iteration_0/"
        self.assetsFolder        = Path(dirs.test_assets) / "test_loop/iteration_0/"
        self.indexPath           = self.assetsFolder / "index_merge_annotations.csv"
        self.newLabelsPath       = self.testFolder / "sampled_images_labels.csv"
        self.newLabelsTranslated = self.testFolder / "sampled_images_labels_translated.csv"

        self.newLabelsIndex = translate_interface_labels_file(self.newLabelsPath)
        # self.newLabelsTemp.to_csv(self.newLabelsTranslated, index=False)

        # self.newLabelsIndex = pd.read_csv(self.newLabelsTranslated)

        ind = IndexManager(self.indexPath)
        ind.merge_annotations(self.newLabelsIndex)

        assert ind.index.loc[21, 'Tags'] == "Duto-Evento-Flange"


    def test_tear_down(self):
        self.testFolder         = Path(dirs.test) / "test_loop/iteration_0/"
        sh.rmtree(self.testFolder)