# Videos of the following formats are found in the data source
videoFormats = ['wmv', 'mpg', 'vob', 'avi', 'VOB']

# Fields of each entry in Index DataFrames
# Field documentation
'''
    'VideoPath':           Filepath of source video. Relative to video dataset folder, ex 20170724_FTP83G_Petrobras.
    'Report':              Source video's Report. Usually first part of VideoPath. Ex 'CIMRL10-676'.
    'DVD':                 Source video's DVD number, if any. Can be an integer or None. Usually second part of VideoPath. Ex '2'.
    'VideoName':           Source video's name, including extension. Usually last part of VideoPath. Ex 'Dive 420 16-02-24 19.32.32_C1.wmv'.
    'FrameTime':           Video time associated with frame, according from Opencv reading of video's metadata.
    'AbsoluteFrameNumber': Frame number according from Opencv reading of video's metadata. Present if the frame comes from a interval based extraction.
    'EventId':             Source Csv event number, if the frame comes from a csv annotated video. Dependent on extraction process.
    'RelativeFrameNumber': Frame number if the frame comes from a csv annotated video.
    'Tags':                Tags associated with frame. Come from an annotation or flagging. It is a string composed of tags separated by '-' character. Would rather be a true list.
    'FramePath':           Frame filepath on disk relative to project folder. Should follow format '<Report>--DVD-<DVD>--<VideoName(w/o ext)>.jpg'
    'FrameName':           Frame file name. Last part of FramePath.
    'OriginalDataset':     Identification of frame's source. Explains from where the entry was obtained.
    'OriginalFramePath':   Original frame location, before it was copied to dataset folder. Likely another dataset folder.
'''
# Field list
indexEntryColumns = [
            'VideoPath',
            'Report',
            'DVD',
            'VideoName',
            'EventId',
            'FrameTime',
            'AbsoluteFrameNumber',
            'RelativeFrameNumber',
            'Tags',
            'FramePath',
            'FrameName',
            'OriginalDataset',
]

unlabeledDatasetName = "unlabeled_images"

reportList = [
            "CIMRL10-676",
            "FAmls16-119",
            "FSll16-224",
            "GHmls16-263",
            "ROBI12-417",
            "SVTab17-001",
            "TCOmll15-017",
            "TCOpm16-140",
            "TCOPM16-160",
            "TVILL16-054",
]
