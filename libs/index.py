import os
import shutil
import pandas       as pd
import numpy        as np
from tqdm           import tqdm
from   datetime     import datetime
from   pathlib      import Path
from   glob         import glob

import libs.dirs    as dirs
from libs.utils     import file_hash


def move_files_routine(source, destination):
    '''
        Wrapper function for shutil.copy2().
        Arguments: source, destination filepaths
        Returns: True if operation was successful and False if it failed.
    '''
    if os.path.isfile(source):
        shutil.copy2(source, destination)
        return True
    else:
        #     print("Source file not found:\n", source)
        return False


class IndexManager:
    def __init__(self, path=dirs.index, destFolder='auto', verbose=True):
        self.path               = Path(path)
        self.indexExists        = False
        self.bkpFolderName      = "index_backup"
        self.destFolder         = destFolder
        self.verbose            = verbose

        self.duplicates_count   = 0
        self.new_entries_count  = 0
        self.originalLen        = 0

        # Get date and time for index and folder name
        self.date = datetime.now()

        self.validate_path()


    def get_index_len(self):
        ''' Returns integer representing number of entries in index, 
        or None, if index doesn't exists.'''
        if self.indexExists:
            return self.index.shape[0]
        else:
            return 0


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
            newFramePath = str(self.newEntryDf.loc[0, 'FrameName'])

            self.newEntryDf['OriginalFramePath'] = self.newEntryDf['FramePath']
            self.newEntryDf['FramePath']         = newFramePath

            if self.indexExists:
                if self.check_duplicates() == True:
                    # If duplicate, merge data with existing entry
                    self.duplicates_count += 1
                else:
                    # If not duplicate, append to existing df
                    self.index = self.index.append(
                                            self.newEntryDf, sort=False,
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
                    "Adding entry from list unsupported for existing index.\
                    Please backup and delete current index and try again."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    )
            numEntries = np.shape(newEntry)[0]

            # If index does not exist, proceed
            self.newEntryDf = pd.DataFrame(newEntry)

            # Save new frame path as FramePath and the old one as OriginalFramePath
            for i in range(numEntries):
                # TODO: Guarantee that FramePath is only formatted outside IndexManager
                # frameName = self.newEntryDf.loc[i, 'FrameName']
                # newFramePath = frameName

                self.newEntryDf.loc[i, 'OriginalFramePath'] = self.newEntryDf.loc[i, 'FramePath']
                # TODO: Guarantee that FramePath is only modified in move_files, when the files are
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


    def write_index(self, auto_path=True, prompt=True):
        '''
            Create a backup of old index and write current index DataFrame to a csv file.
            auto_path == True appends date and time to index path
        '''
        if prompt:
            print("\n\nReally write index to file?\nPress any key to continue, Ctrl-C to cancel.\n")
            input()

        # Create destination folder
        dirs.create_folder(self.path.parent)

        self.make_backup()

        if auto_path == True:
            newName = str(self.path.stem) +"_{}-{}-{}_{}-{}-{}".format(self.date.year, self.date.month,\
             self.date.day, self.date.hour, self.date.minute, self.date.second)

            self.indexPath = self.path.with_name( newName + str(self.path.suffix))
        else:
            self.indexPath = self.path

        self.index.to_csv(self.indexPath, index=False)
        self.report_changes()


    def move_files(self, destFolder='auto', write=True):
        '''
            Try to move all files in index to a new folder specified by destFolder input.
        '''
        assert self.indexExists, "Index does not exist. Cannot move files."

        self.destFolder = destFolder

        if self.destFolder == 'auto':
            self.destFolder = Path(dirs.dataset + "compiled_dataset_{}-{}-{}_{}-{}-{}".format(
                              self.date.year, self.date.month, self.date.day,
                              self.date.hour, self.date.minute, self.date.second))

        dirs.create_folder(self.destFolder, verbose=True)

        print("Moving {} files.".format(self.index.shape[0]))

        f = lambda x: self.destFolder / x
        self.frameDestPaths = self.index.loc[:, 'FrameName'].apply(f)

        self.moveResults = list(map(move_files_routine, self.index.loc[:, 'OriginalFramePath'], self.frameDestPaths))

        for i in range(self.get_index_len()):
            self.index.loc[i, "OriginalFramePath"] = self.index.loc[i, "FramePath"].copy()
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
                    self.destFolder,
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
        self.tagList = []
        f = lambda x: self.tagList.extend(x.split('-'))

        self.index['Tags'].apply(f)
        self.tagList = list(dict.fromkeys(self.tagList))

        return self.tagList


    def append_tag(self, entryIndex, newTag):
        self.index.at[entryIndex, 'Tags'] += "-"+newTag
        print(self.index.at[entryIndex, 'Tags'])


    def get_video_hash_list(self):
        ''' 
            Returns DataFrame with columns 'VideoPath', containing all unique video paths in the index,
            and 'VideoHash', containing the respective file hashes.
        '''
        if self.indexExists:
            self.hashDf = pd.DataFrame({'VideoPath': self.get_video_path_list()})

            numVideos = self.hashDf.shape[0]
            for i in tqdm(range(numVideos)):
                self.hashDf.loc[i, 'VideoHash'] = file_hash(self.hashDf.loc[i, 'VideoPath'])
        else:
            self.hashDf = None
        
        return self.hashDf
