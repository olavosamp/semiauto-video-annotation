import os
import shutil
import time
from copy                   import copy
import pandas               as pd
import numpy                as np
from   tqdm                 import tqdm
from   datetime             import datetime
from   pathlib              import Path
from   glob                 import glob

import libs.dirs            as dirs
import libs.utils           as utils
import libs.dataset_utils   as dutils
# from libs.utils             import *
# from libs.dataset_utils     import *


def hyphenated_string_to_list(hyphenString):
    return hyphenString.split("-")


class IndexManager:
    def __init__(self, path=dirs.index+"main_index.csv", destFolder='auto', verbose=True):
        self.path               = Path(path)
        self.indexExists        = False
        self.bkpFolderName      = "index_backup"
        self.imagesDestFolder   = destFolder
        self.verbose            = verbose

        self.duplicates_count   = 0
        self.new_entries_count  = 0
        self.originalLen        = 0

        # Get date and time for index and folder name
        self.date = datetime.now()

        self.validate_path()


    def get_index_len(self):
        '''
            Returns integer representing number of entries in index, 
            or None, if index doesn't exists.
        '''
        if self.indexExists:
            return self.index.shape[0]
        else:
            return None


    def get_video_path_list(self):
        ''' Returns list with unique videos in the dataset.
        '''
        if self.indexExists:
            return list(dict.fromkeys(self.index['VideoPath']))
        else:
            return []


    def validate_path(self):
        if self.path.suffix == ".csv":
            # Check for csv files matching filename in self.path
            pathList = list(self.path.parent.glob("*"+str(self.path.stem).strip()+"*.csv"))

            if len(pathList) > 0:
                try:
                    # Check if index DataFrame exists and is non-empty
                    self.index = pd.read_csv(pathList[0])
                    self.originalLen = self.index.shape[0]

                    if self.originalLen > 0:
                        self.indexExists = True
                except pd.errors.EmptyDataError:
                    pass
        else:
            raise ValueError("Invalid index path.")


    @staticmethod
    def get_new_frame_path(entry):
        if entry.loc[0, 'DVD'] is None:
            # Put a DVD-X indicator in frame path
            newFramePath = "--".join(
                [entry.loc[0, 'Report'],
                entry.loc[0, 'FrameName']])
        else:
            # Don't put any DVD indicator in path
            newFramePath = "--".join(
                [entry.loc[0, 'Report'],
                'DVD-'+str(entry.loc[0, 'DVD']),
                entry.loc[0, 'FrameName']])
        return newFramePath


    def add_entry(self, newEntry):
        '''
            Adds new entry to index by appending new entries to the existing DataFrame or
            creating a new DataFrame.

            Arguments:
                newEntry: Single entry or list of entries. Each entry is a Dict of lists. Keys are data columns, values are lists containing
                          the data
        '''
        # If input is a single entry
        if type(newEntry) is dict:
            self.newEntryDf = pd.DataFrame.from_dict(newEntry)

            # Save new frame path as FramePath and the old one as OriginalFramePath
            # TODO: Guarantee that FramePath is only formatted outside IndexManager
            
            # If this is a new entry, OriginalFramePath and FramePath will be the same
            # FramePath should only be changed if the file is actually moved
            self.newEntryDf['OriginalFramePath'] = self.newEntryDf['FramePath']
            self.newEntryDf['FramePath']         = str(self.newEntryDf.loc[0, 'FrameName'])

            if self.indexExists:
                if self.check_duplicates() == True:
                    # If duplicate, merge data with existing entry
                    self.duplicates_count += 1
                else:
                    # If not duplicate, append to existing df
                    self.index = self.index.append(self.newEntryDf, sort=False,
                                                    ignore_index=False).reset_index(drop=True)
                    self.new_entries_count += 1
            else:
                # Create df with new entry
                self.index              = self.newEntryDf.copy()
                self.indexExists        = True
                self.new_entries_count += 1

        # If input is a list of entries
        elif type(newEntry) is list:
            if self.indexExists:
                raise ValueError(
                    "Adding entry from list unsupported for existing index.\n\
                    Please backup and delete current index and try again.")
            numEntries = np.shape(newEntry)[0]

            # If index does not exist, proceed
            self.newEntryDf = pd.DataFrame(newEntry)

            # Save new frame path as FramePath and the old one as OriginalFramePath
            for i in range(numEntries):
                # TODO: Guarantee that FramePath is only formatted outside IndexManager
                # frameName = self.newEntryDf.loc[i, 'FrameName']
                # newFramePath = frameName

                self.newEntryDf.loc[i, 'OriginalFramePath'] = self.newEntryDf.loc[i, 'FramePath']
                # TODO: Guarantee that FramePath is only modified in move_files when the files are
                #  actually moved
                # self.newEntryDf.loc[i, 'FramePath']         = newFramePath

            self.index              = self.newEntryDf.copy()
            self.indexExists        = True

            # TODO: Fix duplicates verification. Decide what to do when there are dups in list add
            # dupIndex = self.index.duplicated(subset="FramePath")
            # # Obs: this is not the same check used in check_duplicates
            # dupDf = self.index.loc[dupIndex, :]
            # print(dupDf)
            # print(dupDf.shape)
            # print(np.shape(dupIndex))
            # print(np.sum(dupIndex))
            # exit()

            # if self.check_duplicates():
            #     raise ValueError("Duplicates entries found. Cannot process.")
            self.new_entries_count += numEntries

        return self.newEntryDf


    def check_duplicates(self):
        '''
            Check for duplicated index entries.

            Duplicate criteria:
                Same Report, DVD and FrameName field.

        '''
        mask            = np.equal(self.index['FramePath'], self.newEntryDf['FramePath']).values
        dupNamesIndex   = np.squeeze(np.argwhere(mask == True)).tolist()

        if self.verbose:
            if np.size(dupNamesIndex) > 0:
                print("Duplicate names: ", dupNamesIndex)

        if np.size(dupNamesIndex) >= 1:
            # There are duplicate entries (there should be only 1)
            if np.size(dupNamesIndex) >= 2:
                raise ValueError("Found multiple duplicate entries. Duplicate check should only and always be run following a new addition.")
            else:
                baseIndex   = np.ravel(dupNamesIndex)[0]
                # print("i: ", baseIndex)
                # print("\nExisting entry: \n",   self.index[['FramePath', 'OriginalDataset']].loc[[baseIndex]])
                # print("\nNew entry: \n",        self.newEntryDf[['FramePath', 'OriginalDataset']])
                # input()

                # Get duplicate and existing Tags list
                newTags     = self.index.loc[baseIndex, 'Tags'].split("-")
                newTags.extend(self.newEntryDf.loc[0, 'Tags'].split("-"))

                # Get duplicate and existing OriginalDataset list
                newDataset  = self.index.loc[baseIndex, 'OriginalDataset'].split("-")
                newDataset.extend(self.newEntryDf.loc[0,'OriginalDataset'].split("-"))

                # Get unique tags and OriginalDataset
                newTags    = list(dict.fromkeys(newTags))
                newDataset = list(dict.fromkeys(newDataset))

                # Save new fields with "-" as separator
                self.index.loc[baseIndex, 'Tags'] = "-".join(newTags)
                self.index.loc[baseIndex, 'OriginalDataset'] = "-".join(newDataset)

                return True

        else: # Entry is not duplicate
            return False


    def make_backup(self):
        '''
            Moves any index files in destination folder to a backup folder.
        '''
        # Create backup folder
        dirs.create_folder(self.path.parent / self.bkpFolderName)

        existingIndex = self.path.parent.glob("*index*.csv")
        for entry in existingIndex:
            entry = Path(entry)
            newPath = self.path.parent / self.bkpFolderName / entry.name

            # Check if dest path already exists
            # If True, create a new path by appending a number at the end
            fileIndex = 2
            while newPath.is_file():
                newPath = self.path.parent / self.bkpFolderName / (entry.stem + "_" + str(fileIndex) + entry.suffix)
                fileIndex += 1

            os.rename(entry, newPath)


    def write_index(self, dest_path='auto', make_backup=True, prompt=True):
        '''
            Create a backup of old index and write current index DataFrame to a csv file.
            auto_path == True appends date and time to index path
        '''
        if prompt:
            print("\n\nReally write index to file?\nPress any key to continue, Ctrl-C to cancel.\n")
            input()

        if make_backup:
            self.make_backup()

        if dest_path == 'auto':
            newName = str(self.path.stem) + "_" + utils.get_time_string(self.date)
            self.indexPath = self.path.with_name( newName + str(self.path.suffix))
        else:
            self.indexPath = Path(dest_path)

        # Create destination folder
        dirs.create_folder(self.indexPath.parent)

        self.index.to_csv(self.indexPath, index=False)
        self.report_changes()


    def copy_files(self, imagesDestFolder='auto', write=False, mode='copy'):
        '''
            Try to move all files in index to a new folder specified by destFolder input.
        '''
        assert self.indexExists, "Index does not exist. Cannot move files."

        self.imagesDestFolder = imagesDestFolder

        if self.imagesDestFolder == 'auto':
            self.imagesDestFolder = Path(dirs.dataset + "compiled_dataset_{}-{}-{}_{}-{}-{}".format(
                              self.date.year, self.date.month, self.date.day,
                              self.date.hour, self.date.minute, self.date.second))

        dirs.create_folder(self.imagesDestFolder, verbose=True)

        print("Copying {} files.".format(self.index.shape[0]))

        def _add_folder_path(x): return self.imagesDestFolder / x
        self.frameDestPaths = self.index.loc[:, 'FrameName'].apply(_add_folder_path)

        # Select copy or move mode
        if mode == 'copy':
            self.moveResults = list(map(utils.copy_files, self.index.loc[:, 'OriginalFramePath'], self.frameDestPaths))
        else:
            raise NotImplementedError

        for i in range(self.get_index_len()):
            self.index.loc[i, "OriginalFramePath"] = copy(self.index.loc[i, "FramePath"])
            self.index.loc[i, "FramePath"] = self.frameDestPaths[i]

        if write:
            self.write_index(prompt=False)

        # Report results
        print(
            "Found {} files.\n\
            Moved {} files to folder\n\
            {}\
            \n{} files were not found."
            .format(len(self.moveResults), sum(self.moveResults),
                    self.imagesDestFolder,
                    len(self.moveResults) - sum(self.moveResults)))
        return self.moveResults


    def report_changes(self):
        print(
            "Original Index had {} entries.\nNew Index has {} entries.".format(
                self.originalLen, self.index.shape[0]))

        print(
            "\nProcessed {} entries. Added {} and merged {} duplicates.\nSaved index to \n{}\n"
            .format(self.new_entries_count + self.duplicates_count,
                    self.new_entries_count, self.duplicates_count,
                    self.indexPath))


    def get_unique_tags(self):
        '''
            Get unique tags over the entire Index.
            Return a list of tags.
        '''
        def _hyphenated_string_to_list(x): self.tagList.extend(x.split('-'))
        self.tagList = []

        self.index['Tags'].apply(_hyphenated_string_to_list)
        self.tagList = list(dict.fromkeys(self.tagList))
        return self.tagList


    def append_tag(self, entryIndex, newTag):
        '''
            Add newTag input to Tag column at entryIndex. newTag is appended with
            '-' separator.

            newTag: String or list of strings to be added to existing Tag column.
            entryIndex: target row index of self.index where the new tags will be added.
        ''' 
        def _func_append_tag(tagArg):
            # Check if argument is a string, then append to append_tag::tag.
            if isinstance(tagArg, str):
                # self.tagsToAppend.append(tagArg)
                self.tagsToAppend.extend(tagArg.split('-'))
            else:
                raise TypeError("Argument must be a string or list of strings.")

        self.tagsToAppend = hyphenated_string_to_list(self.index.loc[entryIndex, 'Tags'])

        # Check if newTag is a list or a string
        if isinstance(newTag, list):
            for t in newTag:
                _func_append_tag(t)
        else:
            _func_append_tag(newTag)
        
        self.tagsToAppend = list(dict.fromkeys(self.tagsToAppend))
        self.index.loc[entryIndex, 'Tags'] = "-".join(self.tagsToAppend)
        
        # Delete unlabeled tag, as it is now labeled
        self.delete_tag(entryIndex, 'unlabeled')


    def delete_tag(self, entryIndex, targetTag, raise_error=False):
        '''
            Delete one instance of targetTag tag of self.index[entryIndex] tag list.
        '''
        tagList = hyphenated_string_to_list(self.index.loc[entryIndex, 'Tags'])
        if targetTag in tagList:
            tagList.remove(targetTag)
        elif raise_error:
            # Raises error if tag not found and raise_error flag is set to True
            raise ValueError("Target tag not found at desired entry.")

        self.index.loc[entryIndex, 'Tags'] = "-".join(tagList)


    def get_video_hash_list(self):
        ''' 
            Returns DataFrame with columns 'VideoPath', containing all unique video paths in the index,
            and 'VideoHash', containing the respective file hashes.
        '''
        if self.indexExists:
            self.hashDf = pd.DataFrame({'VideoPath': self.get_video_path_list()})

            numVideos = self.hashDf.shape[0]
            for i in tqdm(range(numVideos)):
                self.hashDf.loc[i, 'VideoHash'] = utils.file_hash(self.hashDf.loc[i, 'VideoPath'])
        else:
            self.hashDf = None
        
        return self.hashDf


    # def estimate_hashes(self, videoFolder):
    #     '''
    #         For each entry, using the value in VideoPath, estimate the best correspondent LocalisedVideoPath
    #         through a regex comparison, compute its MD5 file hash and save it in a new columns.

    #         Creates a new HashMD5 column for the index DataFrame.
    #     '''
    #     self.videoFolder = videoFolder
    #     self.videoList   = get_file_list(videoFolder, ext_list=commons.videoFormats, remove_dups=True)

    #     self.hashTable   = pd.read_csv(dirs.hashtable)

    #     assert len(self.videoList) == self.hashTable.shape[0], "Video hash table must have an entry for every video in VideoFolder.\nHash table has {} entries and VideoFolder has {}".format(
    #         len(self.videoList, self.hashTable.shape[0]))

    #     hashList = make_video_hash_list(self.index.loc[:, "VideoPath"].values)
    #     # TODO: Change check_duplicates to use file hashes instead of convoluted string field comparisons

    #     # TODO: Add matching entries to HashMD5 columns; Treat non matching VideoPaths;


    def compute_frame_hashes(self, reference_column='FramePath'):
        '''
            Compute MD5 hashes for every frame in the index. The reference column
            is given by the input string reference_column.
            
            Hashes are saved to new column 'FrameHash'.
        '''
        if self.indexExists:
            start = time.time()

            print("Calculating hashes of {} images...".format(self.get_index_len()))
            self.index["FrameHash"] = self.index.loc[:, reference_column].apply(utils.file_hash)

            elapsedTime = time.time() - start
            return elapsedTime
        else:
            raise ValueError("Index does not exists.")


    def check_files(self):
        '''
            Verifies integrity of the index by checking if every entry FramePath value
             points to an existing image.
        '''
        print("\nVerifying file path integrity.")
        fileCheck = self.index.loc[:, 'FramePath'].apply(utils.file_exists)

        mask = np.logical_not(fileCheck)
        notFound = np.extract(mask, self.index.loc[:, 'FramePath'])
        
        if fileCheck.sum() < fileCheck.shape[0]:
            for elem in notFound:
                print(Path(elem))
            print("\nThe above entries' FramePath values did not point to a valid file.")

        print("\nTotal entries: ", fileCheck.shape[0])
        print("Files not found: ", len(notFound))


    def merge_annotations(self, newLabelsIndex):
        '''
            Add new annotations in newLabels interface-generated csv to existing annotated 
            images csv, existingIndex.

            existingIndex path will be inferred through newLabels path, which must follow
                the template "<loop_folder>/iteration_#/sampled_images_labels.csv".
            The function will search for a csv file at <loop_folder> and, if found, assume
                it is the existingIndex. If none is found, a new index will be created.

            The standard name for existingIndex is labeled_images_index.csv

            Every entry in newLabels must be present in existingIndex, if it exists, or 
            in a "iteration_#/sampled_images_iteration_#.csv" file.
        '''
        newLabelLen = newLabelsIndex.shape[0]
        # print(tagList)
        # input()
        self.index.set_index('FrameHash', drop=False, inplace=True)
        # print("Before new tags append")
        # print(self.index.loc["bb310a3b9bb72b81326cd70c29117c4b", :])

        for i in range(newLabelLen):
            ind     = newLabelsIndex.loc[i, 'FrameHash']
            tagList = newLabelsIndex.loc[i, 'Tags']
            # self.append_tag(ind, tagList[i])
            self.append_tag(ind, tagList)

        # print("\n\nAfter new tags append")
        # print(self.index.loc["bb310a3b9bb72b81326cd70c29117c4b", :])
        # input()
        self.index.reset_index(drop=True, inplace=True)
        # print(self.index.iloc[21, :])
