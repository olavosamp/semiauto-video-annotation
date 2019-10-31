import os
import re
import math
import shutil
import pickle
import hashlib
import subprocess
import numpy     as np
import pandas    as pd
from PIL         import Image
from glob         import glob
from tqdm        import tqdm
from pathlib     import Path

import libs.dirs    as dirs
import libs.commons as commons


## Small lambda functions
def func_make_path(x):   return Path(x)
def func_strip(x):       return Path(str(x).strip())


# Numeric functions
def normalize_array(array):
    maxVal = np.max(array, axis=0)
    minVal = np.min(array, axis=0)
    dif = np.abs(maxVal - minVal)

    return (array - minVal)/(dif)


def get_perfect_square(number, round='down'):
    if round == 'down':
        return int(math.sqrt(number))**2
    else:
        return round(math.sqrt(number))**2


## Pickle functions
def save_pickle(object, filePath):
    with open(filePath, 'wb') as fileHandle:
        pickle.dump(object, fileHandle)
    return 0


def load_pickle(filePath):
    with open(filePath, 'rb') as fileHandle:
        pickledInfo = pickle.load(fileHandle)
    return pickledInfo


## Time and Date
def timeConverter( strTime ):
    '''
        int seconds = timeConverter( string strTime )
        
        Converts HHMMSS input string to integer seconds
    '''
    length = len(strTime)
    if length > 6:
        seconds = 0
    else:
        h = int(strTime[0:2])
        m = int(strTime[2:4])
        s = int(strTime[4:6])

        seconds = s + m*60 + h*3600
    return seconds


def get_time_string(date):
    ''' 
        Argument: datetime object

        Returns:  Formatted string with year, month, day, hour, minute and seconds.
    '''
    timeString = "{:.0f}-{:.0f}-{:.0f}_{:.0f}-{:.0f}-{:.0f}".format(date.year, date.month,\
        date.day, date.hour, date.minute, date.second)
    return timeString


## Index and DataFrames
def subset_column(main_index, frame_hash_list, target_column):
    '''
        Opens the DataFrame main_index, selects entries corresponding to FrameHash indexes in 
        frame_hash_list and return only the column given by target_column.
    '''
    # Select index entries by ImgHash
    main_index = main_index.set_index('FrameHash', drop=False, inplace=False)
    subsetColumn = main_index.loc[frame_hash_list, target_column].copy()

    print(subsetColumn.shape)
    return subsetColumn


## Filepath and string processing
def add_folder_path(path, folder):
    path = folder / Path(path)
    return str(path)


def compute_file_hash_list(file_list, folder=None):
    '''
        Argument:
           file_list: list of strings
                List of valid file paths.
        Returns:
            : list of strings
                List of MD5 hashes.
    '''
    if folder is not None:
        fileLen = len(file_list)
        file_list = list(map(add_folder_path, file_list, [folder]*fileLen))
    return list(map(file_hash, file_list))


def check_empty_file(path):
    if os.path.isfile(path):
        return os.path.getsize(path) > 0
    else:
        return False


def make_videos_hash_list(fileList, filepath_column='FilePath', hash_column="FrameHash", verbose=True):
    '''
        Find and save video paths in a file tree in a list,
         calculate their MD5 hashes and save both lists as a Pandas DataFrame.

        Argument:
            fileList: list of strings. List of paths of the files to hash.
            filepath_column: name of the DataFrame columns where the filepath list
             will be saved.

        Returns:
            table: Pandas DataFrame with two columns: filepath_column and HashMD5.
    '''    
    # hashList = list(map(file_hash, fileList))
    numVideos = len(fileList)
    if verbose:
        print("Processing MD5 hash of {} files...".format(numVideos))
    
    hashList = []
    for i in tqdm(range(numVideos)):
        # print("{}".format(i).ljust(5), "/{} :\n{}".format(numVideos, fileList[i]))
        hashList.append(file_hash(fileList[i]))

    table = pd.DataFrame({str(filepath_column): fileList, hash_column: hashList})
    return table


def file_exists(x):
    return Path(x).is_file()
    # return os.path.isfile(x)


def replace_symbols(stringList, og_symbol="\\", new_symbol="/"):
    '''
        Replaces symbols in strings or string lists.

        Replace all instances of og_symbol with new_symbol.
        Defaults to replacing backslashes ("\\") with foward slashes ("/").
    '''
    def _func_replace(x): return str(x).replace(og_symbol, new_symbol)

    if isinstance(stringList, str):         # Input is string
        return _func_replace(stringList)
    elif hasattr(stringList, "__iter__"):   # Input is list of strings
        return list(map(_func_replace, stringList))
    else:
        raise TypeError("Input must be a string or list of strings.")


def make_path(filePathList):
    '''Makes every entry in a filepath list a pathlib.Path object.'''
    def _path_func(x):  return Path(x)
    return list(map(_path_func, filePathList))


def get_relative_list(fileList, refFolder):
    '''
        Input:   a list of filepaths and a reference folder
        Returns: the input list with every entry relative to the ref folder
    '''
    def func_rel_to(x): return str(Path(x).relative_to(refFolder))
    return list(map(func_rel_to, fileList))


def get_file_list(folderPath, ext_list=['*'], remove_dups=True):
    '''
        Returns list of files in the file tree starting at folderPath.

        Optional argument ext_list defines list of recognized extensions, case insensitive.
        ext_list must be a list of strings, each defining an extension, without dots.
        
        Argument remove_dups should almost always be True, else it will return duplicated entries
        as it searches both upper and lower case versions of all given extensions.
    '''

    # Also search for upper case formats for Linux compatibility
    ext_list.extend([x.upper() for x in ext_list])
    # TODO: Replace this workaround by making a case insensitive search or
    # making all paths lower case before making comparisons (possible?)

    folderPath = replace_symbols(folderPath, og_symbol="\\", new_symbol="/")

    fileList = []
    for format in ext_list:
        globString = folderPath + "/**" + "/*."+format
        globList = glob(globString, recursive=True)
        fileList.extend(globList)
    
    if remove_dups:
        # Remove duplicated entries
        fileList = list(dict.fromkeys(fileList))
    
    fileList = list(map(replace_symbols, fileList))

    return fileList


def remove_video_ts(videoList):
    '''
        Delete VIDEO_TS.VOB DVD headers.
        Returns numpy array without header entries.
    '''
    # Map function
    def matchVideoTS(x): return (str(x).find("VIDEO_TS.VOB") == -1)

    mask      = list(map(matchVideoTS, videoList))                  # Get index mask
    videoList = np.array(videoList)[mask]                           # Make path list into numpy array and
                                                                    # apply index mask
    return list(videoList)


def copy_files(source, destination):
    '''
        copy_files(source, destination)

        Copy file at source to destination path.
    '''
    if os.path.isfile(source):
        dirs.create_folder(Path(destination).parent)
        shutil.copy2(source, destination)
        return True
    else:
        return False
    #     print("Source file not found:\n", source)


def string_list_complement(list1, list2):
    '''
        Arguments:
            list1, list2: Two lists of strings.
        Return value:
            list3: Set complement of the arguments, list1 - list2.
            Only contains elements of list1 that are not in list2.
    '''
    def _compare(path1, path2):
        '''
            Returns True if path1 contains path2, else returns False.
        '''
        path1 = Path(path1)
        path2 = Path(path2)

        pattern = ""
        numParts = len(path2.parts)
        for i in range(numParts-1):
            pattern += str(path2.parts[i]) + ".*"
        pattern += path2.parts[-1]#.replace('.', '\.')
        pattern = str(pattern)
        if re.search(pattern, str(path1)):
            return True
        else:
            return False

    list3 = []
    for elem1 in list1:
        #print("Searching for\n{}\n".format(elem1))
        #input()
        appendFlag = True
        for elem2 in list2:
            #print("{}\n{}\n{}\n".format(elem1, elem2, _compare(elem1, elem2)))
            if _compare(elem1, elem2):
                #print("Labeled video found. Not adding to list.\n")
                appendFlag = False
                break

        if appendFlag:
            list3.append(elem1)
            #print("Labeled video not found for\n{}. Adding to list.\n".format(elem1))
            #print("List size: {}.\n".format(len(list3)))
            #input()

    return list3


def add_ok(pathList):
    '''
        Appends "_OK" to reports created without this termination. Also replaces '\\' with '/'.

        pathList: List of string paths.
    '''
    def _replace(x):
        for report in commons.reportList:
            x = str(x).replace(report+"/", report+"_OK"+"/")    # Must append _OK only to strings without it
            x = x.replace(report+"\\", report+"_OK"+"\\")  # Do it twice for Linux/Windows compatibility
            x = x.replace("\\", "/")
        return x
    return list(map(_replace, pathList))


def file_hash(filePath):
    '''
        Returns md5 file hash string corresponding to file found at filePath.
    '''
    with open(filePath, 'rb') as handler:
        data = handler.read()
        hashedData = hashlib.md5(data).hexdigest()
    return hashedData


## Video and image processing
def convert_video(video_input, video_output):
    print("\nProcessing video: ", video_input)
    print("Saving to : ", video_output)

    destFolder = '/'.join(video_output.split('/')[:-1])
    dirs.create_folder(destFolder)

    cmds = ['ffmpeg', '-i', video_input, video_output]
    subprocess.Popen(cmds)

    print("Video saved to : ", video_output)
    return 0


def color_filter(image, filter='r', filter_strenght=1.5):
    # Select color channel
    if filter == 'r':
        targetChannel = 0
    elif filter == 'g':
        targetChannel = 1
    elif filter == 'b':
        targetChannel = 2
    else:
        raise KeyError("Selected filter does not exist.")

    source = image.split()

    redFilter = source[targetChannel].point(lambda x: x*filter_strenght)

    source[targetChannel].paste(redFilter)
    image = Image.merge(image.mode, source)
    return image


def image_grid(path, targetPath="image_grid.jpg", prediction_index=None, upperCrop=0, lowerCrop=0,
               size_limit=None, show=False, save=True):
    '''
        Creates a square grid of images randomly samples from available files on path.

        path:
            Target images folder path;

        targetPath:
            Path where resulting grid will be saved;

        upperCrop and lowerCrop:
            Number of pixels to be cropped from each composing image. The crops executed
        are horizontal crops and are measured from top to center and bottom to center,
        respectively.
    '''
    targetPath = Path(targetPath)
    files = glob(str(path)+'**'+dirs.sep+'*.jpg', recursive=True)
    numImages         = len(files)
    if size_limit is not None:
        numImages = np.clip(numImages, None, size_limit)
    squareNumImages = get_perfect_square(numImages)

    files = np.random.choice(files, size=squareNumImages, replace=False)

    # TODO: This should be done in an external test file
    # # Create fake predictions DataFrame 
    # prediction_index = pd.DataFrame(files)
    # prediction_index['Prediction'] = np.random.choice([0, 1], size=squareNumImages, p=[0.8, 0.2])

    # Square Grid
    # Side of a square image grid. It will contain side^2 images.
    side = int(math.sqrt(numImages))

    # Image resizing dimension
    imageDim = (300,300)          # (width, height)
    # imageDim = (100,100)

    destDim = (side*imageDim[0], side*(imageDim[1] - lowerCrop - upperCrop))

    im_grid = Image.new('RGB', destDim)
    index = 0
    for j in tqdm(range(0, destDim[1], imageDim[1] - lowerCrop - upperCrop)):
        for i in range(0,destDim[0], imageDim[0]):
            try:
                im = Image.open(files[index])
            except:
                continue

            im = im.resize(imageDim)
            im = im.crop((0, upperCrop, imageDim[0], imageDim[1] - lowerCrop))

            # TODO: Test this properly
            if prediction_index is not None:
                # Apply color filter if image has wrong prediction
                if prediction_index.loc[index, "Prediction"] == 1:
                    im = color_filter(im, filter='r', filter_strenght=3.5)

            im.thumbnail(imageDim)
            im_grid.paste(im, (i,j))
            index += 1

    if save is True:
        dirs.create_folder(targetPath.parent)

        im_grid.save(targetPath)
        print("\nYour image grid is ready. It was saved at {}\n".format(targetPath))
    if show is True:
        im_grid.show()
    return 0
