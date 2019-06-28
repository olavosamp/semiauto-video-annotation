import cv2
import pandas       as pd
import numpy        as np
import datetime
import os.path
from pathlib        import Path
from tqdm           import tqdm

import libs.dirs    as dirs
import libs.commons as commons
from libs.utils     import timeConverter

class GetFrames:
    '''
        Base frame extractor class
    '''
    def __init__(self, destPath, videoFolder=dirs.base_videos, verbose=True, errorLog=True):
        self.destPath       = Path(destPath)
        self.verbose        = verbose
        self.errorLog       = errorLog
        self.estimatedFPS   = False
        self.videoFolder    = videoFolder

        self.videoError   = {'read': False, 'set': False, 'write': False}
        if self.errorLog:
            self.errorCounter = {'read': 0,     'set': 0,     'write': 0}
            self.errorList = []

        self.frameCount  = 0
        self.datasetName = commons.unlabeledDatasetName

        if self.verbose:
            print("\nUsing opencv version: ", cv2.__version__)

        # Create destination folder
        dirs.create_folder(self.destPath)


    def get_video_data(self):
        '''
            videoPath: source video path
        '''
        # Get relative video path from full video path
        self.videoName      = Path(self.videoPath.name)

        # Get Report field
        self.videoReport = self.videoPath.relative_to(self.videoFolder).parts[0]
        # Get DVD field
        dvdIndex = str(self.videoPath).find("DVD-")
        if dvdIndex == -1:
            self.dvd = None
        else:
            self.dvd = str(self.videoPath)[dvdIndex+4]

        try:
        	self.video = cv2.VideoCapture(str(self.videoPath))
        except:
            print("\nError opening video:\n")
            cv2.VideoCapture(str(self.videoPath))

        self.frameRate = self.video.get(cv2.CAP_PROP_FPS)
        if self.frameRate == 0:
            self.frameRate = 25  # Default frame rate is 30 or 25 fps
            self.estimatedFPS = True

        self.totalFrames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

        if self.dvd != None:
            self.videoFolderPath = self.destPath / self.videoReport / ("DVD-" + self.dvd) / Path(self.videoName.stem)
        else:
            self.videoFolderPath = self.destPath / self.videoReport / Path(self.videoName.stem)
        dirs.create_folder(self.videoFolderPath)
        return self.video


    def get_frames(self):
        self.frameEntryList = []
        self.totalFrames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.videoTime   = self.totalFrames/self.frameRate
        # if self.verbose:
        #     print("Total video time:")
        print("Video time: ", str(datetime.timedelta(seconds=self.videoTime)))

        self.timePos    = 0
        self.frameCount = 0

        # Actual frame extraction function
        self._routine_get_frames()

        print(self.videoError)
        print("{} frames captured.".format(self.frameCount))
        return self.frameEntryList


    def reset_error_flags(self):
        # Reset video error flags
        self.videoError["set"]   = True
        self.videoError["read"]  = True
        self.videoError["write"] = True



class GetFramesCsv(GetFrames):
    '''
        csvPath:  reference csv filepath
        destPath: dataset destination folder
        interMin: minimum frame capture interval, in seconds
        interMax: maximum frame capture interval, in seconds
    '''
    def __init__(self, csvPath, destPath='./images/', videoFolder=dirs.base_videos, interMin=0.8, interMax=20,
     verbose=True, errorLog=True, ignorePathError=False):
        super().__init__(destPath, videoFolder=videoFolder, verbose=verbose)
        self.ignorePathError = ignorePathError
        self.csvPath         = csvPath
        self.interMin        = interMin
        self.interMax        = interMax

        # Get csv data
        if self.csvPath != None:
            self.csvData   = self.get_csv_data()
        else:
            raise NameError("CSV filepath not defined.")


    # def validate_csv_path(self):
    #     # Check if csv path exists


    def get_csv_data(self):
        # CSV Data is a pandas DataFrame of the csv file
        self.csvData = pd.read_csv(self.csvPath, dtype=str)

        # Validate video paths
        # Assumes that there can be many videopaths in one csv
        # note: iterrows does not preserve dtypes;
        #       don't modify dataframe while iterating.
        for index, row in self.csvData.iterrows():
            checkPath = self.videoFolder+row['VideoName']
            if os.path.isfile(checkPath) == False:
                print("\n", checkPath, "\n")
                raise FileNotFoundError("Csv points to a video that doesn't exist.")
                # TODO: Check if video path has extension; try to add an extension

        # Assumes there can be only one videopath in the entire csv
        # self.videoPath    = self.videoFolder+self.csvData.loc[0, 'VideoName']
        self.videoName    = self.csvData.loc[0, 'VideoName']
        return self.csvData


    def get_capture_interval(self):
        self.interval = 20*(self.eventTime*self.numEntries/self.totalFrames)#*1000
        self.interval = np.clip(self.interval, self.interMin, self.interMax)

        return self.interval


    def get_filename(self):
        videoNameClean = self.videoName.replace("/", "--")[:-1][:-4]
        self.frameName = videoNameClean+" ID{} FRAME{} {}.jpg".format(self.eventId,
                        self.eventFrames, self.eventClass)

        self.framePath = self.videoFolderPath+self.frameName


    def get_frames(self):
        # self.totalFrames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        # self.videoTime   = self.totalFrames/self.frameRate
        # if self.verbose:
        #     print("Total video time:")
        # print(str(datetime.timedelta(seconds=self.videoTime)))

        # Actual frame extraction function
        self._routine_get_frames()

        print(self.videoError)
        print("{} frames captured.".format(self.frameCount))


    def _routine_get_frames(self):
        ''' Get frames from video using a csv file as reference'''

        self.video = self.get_video_data()
        print("\nframeRate: {:.2f}".format( self.frameRate))
        # print("totalFrames:", self.totalFrames)
        print()


        # if self.totalFrames == 0:
            # TODO: GET TOTAL FRAMES AND VIDEO TIME SOME WAY

        # self.totalFrames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.videoTime   = self.totalFrames/self.frameRate
        self.numEntries = self.csvData.shape[0]
        # print("\nFound ", self.numEntries, "lines in csv.\n")

        self.frameCount = 0
        for self.eventNum in range(self.numEntries):
            self.eventFrames= 0
            self.eventStart = timeConverter(self.csvData.loc[self.eventNum,'StartTime'])#*1000
            self.eventEnd   = timeConverter(self.csvData.loc[self.eventNum,'EndTime'])#*1000
            self.eventClass = self.csvData.loc[self.eventNum,'Class']
            self.eventId    = self.csvData.loc[self.eventNum, 'Id']

            self.eventTime = self.eventEnd - self.eventStart

            if self.eventClass not in commons.classes:
                print("\n\nError: Proposed class is not in accepted classes list.\nSkipping entry.\n\n")
                continue

            self.interval  =  self.get_capture_interval()

            self.timePos   = self.eventStart
            self.timeLimit = self.eventEnd

            print("\nEvent ", self.eventId)
            print("timeStart: ", self.eventStart)
            print("interval: ", self.interval)
            print("timeEnd: ", self.eventEnd)

            while self.timePos < self.timeLimit:
                # if self.verbose:
                    # print("Frame ", self.eventFrames)
                    # print("Time: ", self.timePos)

                self.videoError['set'] = self.video.set(cv2.CAP_PROP_POS_MSEC, self.timePos*1000)

                # frameNum is Absolute frame number
                # frameCount is the number of frames extracted from the video so far
                self.frameNum = self.video.get(cv2.CAP_PROP_POS_FRAMES)

                self.videoError['read'], self.frame = self.video.read()

                # Check for read or seek erros before writing frame
                if self.videoError['set'] and self.videoError['read']:
                    self.get_filename()
                    self.videoError['write'] = cv2.imwrite(self.framePath, self.frame)
                    self.eventFrames += 1

                self.timePos     += self.interval

                if self.errorLog:
                    for key in self.videoError.keys():
                        if self.videoError[key] == False:
                            self.errorCounter[key] += 1
                            self.errorList.append((self.framePath, key))

                # Print errors
                if self.verbose:
                    if (self.videoError['set'] == False) or (self.videoError['read'] == False) or (self.videoError['write'] == False):
                        print("\n!!! Error !!!\n\
                        Set  : {}\n\
                        Read : {}\n\
                        Write: {}\n".format(self.videoError['set'],self.videoError['read'],self.videoError['write']))

                self.reset_error_flags()

            print("Captured frames: {}\n".format( self.eventFrames))
            self.frameCount += self.eventFrames



class GetFramesFull(GetFrames):
    def __init__(self, videoPath, videoFolder=dirs.base_videos, destPath='./images/', interval=5, interMin=0.8, interMax=20, verbose=True):
        super().__init__(destPath, videoFolder=videoFolder, verbose=verbose)
        self.videoPath  = Path(videoPath)
        self.interval   = interval
        self.interMin   = interMin
        self.interMax   = interMax


        # Validate video path and file
        self.video = self.get_video_data()

        self.validate_interval()


    def validate_interval(self):
        self.interval = np.clip(self.interval, 1/self.frameRate, None)
        self.interMin = np.clip(self.interMin, 1/self.frameRate, None)
        self.interMax = np.clip(self.interMax, 1/self.frameRate, None)


    def get_filename(self):
        self.frameName = str(self.videoName)+ " FRAME {}.jpg".format(self.frameNum)
        # self.frameName = "--".join([self.videoReport, 'DVD-'+str(self.dvd), str(self.videoName)+ " FRAME {}.jpg".format(self.frameNum))
        self.filePath = self.videoFolderPath / self.frameName

        return str(self.filePath)


    def _routine_get_frames(self):
        if self.verbose:
            print("Full video frame capture")

        self.timeLimit = self.videoTime
        while self.timePos < self.timeLimit:
            if self.verbose:
                print("Frame ", self.frameCount)

            # self.frameEntry = dict()
            self.videoError['set'] = self.video.set(cv2.CAP_PROP_POS_MSEC, self.timePos*1000)

            self.frameNum = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            self.videoError['read'], self.frame = self.video.read()

            self.videoError['write'] = cv2.imwrite(self.get_filename(), self.frame)

            # if self.verbose:
            #     print("Frame number: ", int(self.frameNum), "  Frame time: ", self.timePos)


            self.frameEntry = {
            'VideoPath':            [self.videoPath],
            'Report':               [self.videoReport],
            'DVD':                  [self.dvd],
            'VideoName':            [self.videoName],
            'EventId':              [None],
            'FrameTime':            [self.timePos],
            'AbsoluteFrameNumber':  [self.frameNum],
            'RelativeFrameNumber':  [None],
            'Tags':                 ["unlabeled"],
            'FramePath':            [self.filePath],
            'FrameName':            [self.filePath.name],
            'OriginalDataset':      [self.datasetName]
            }

            # print("\n")
            # print('VideoPath ', self.videoPath)
            # print('Report ', self.videoReport)
            # print("dvd: ", self.dvd)
            # print("videoname: ", self.videoName)
            # # print('EventId: ', self.eventId)
            # print('FrameTime: ', self.timePos)
            # print('AbsoluteFrameNumber: ', self.frameNum)
            # # print('RelativeFrameNumber: ', self.relFrame)
            # # print('Tags: ', self.tags)
            # print('FramePath: ',       self.filePath)
            # print('FrameName: ',       self.filePath.name)
            # print("OriginalDataset: ", self.datasetName)
            # input()

            self.frameEntryList.append(self.frameEntry)

            self.timePos    += self.interval
            self.frameCount += 1
