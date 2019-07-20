import os
import re
import math
import shutil
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


## Filepath and string processing
def get_time_string(date):
    ''' Argument: datetime object
        Returns:  Formatted string with year, month, day, hour, minute and seconds.
    '''
    timeString = "{}-{}-{}_{}-{}-{}".format(date.year, date.month,\
        date.day, date.hour, date.minute, date.second)
    return timeString


def copy_files(source, destination):
    if os.path.isfile(source):
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
            list3: Set complement of the arguments, list1 - list2. Contains elements of list1 that are not in list2.
    '''
    def _compare(path1, path2):
        '''
            Returns True if path1 contains path2, else returns False.
        '''
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
        appendFlag = False
        for elem2 in list2:
            #print("{}\n{}\n{}\n".format(elem1, elem2, _compare(elem1, elem2)))
            if _compare(elem1, elem2):
                #print("Labeled video found. Not adding to list.\n")
                appendFlag = True
                break

        if not(appendFlag):
            list3.append(elem1)
            #print("Labeled video not found for\n{}. Adding to list.\n".format(elem1))
            #print("List size: {}.\n".format(len(list3)))
            #input()

    return list3


def add_ok(pathList):
    '''
        Appends "_OK" to reports created without this termination.

        pathList: List of string paths.
    '''
    def _replace(x):
        for report in commons.reportList:
            x = str(x).replace(report+"/", report+"_OK"+"/")    # Must guarantee to only append _OK to strings without it
            x = str(x).replace(report+"\\", report+"_OK"+"\\")  # Do it twice for Linux/Windows compatibility
        return x
    return list(map(_replace, pathList))


def file_hash(filePath):
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


def get_perfect_square(number, round='down'):
    if round == 'down':
        return int(math.sqrt(number))**2
    else:
        return round(math.sqrt(number))**2


def timeConverter( strTime ):
    # int seconds = timeConverter( string strTime )
    # Converts HHMMSS input string to integer seconds
    #
    length = len(strTime)
    if length > 6:
        seconds = 0
    else:
        h = int(strTime[0:2])
        m = int(strTime[2:4])
        s = int(strTime[4:6])

        seconds = s + m*60 + h*3600
    return seconds


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


def image_grid(path, targetPath="image_grid.jpg", upperCrop=0, lowerCrop=0, show=False, save=True):
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
    squareNumImages = get_perfect_square(numImages)

    files = np.random.choice(files, size=squareNumImages, replace=False)

    # Create fake predictions DataFrame
    predictions = pd.DataFrame(files)
    predictions['Prediction'] = np.random.choice([0, 1], size=squareNumImages, p=[0.8, 0.2])

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
            im = Image.open(files[index])

            im = im.resize(imageDim)
            im = im.crop((0, upperCrop, imageDim[0], imageDim[1] - lowerCrop))

            # Apply color filter if image has wrong prediction
            if predictions.loc[index, "Prediction"] == 1:
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
