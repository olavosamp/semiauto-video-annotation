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
    'FramePath':           Frame filepath on disk relative to project folder. Should follow format '<Report>--DVD-<DVD>--<VideoName(w/o ext)>.jpg' or '<Report>--<VideoName(w/o ext)>.jpg', if there is no associated DVD.
    'FrameName':           Frame file name. Last part of FramePath.
    'OriginalDataset':     Identification of frame's source. Indicates from where the entry was obtained.
    'OriginalFramePath':   Original frame location, before it was copied to dataset folder. Likely another dataset folder.
    'HashMD5':             MD5 hash of the source video.
    'FrameHash':           MD5 hash of the image file.
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
            'HashMD5',
            'FrameHash'
]

unlabeledDatasetName = "unlabeled_dataset"

reportList = [  # Report list without _OK suffix
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

classes = { # This translation table follows the non-standard format described below.
            # Standard class name:  [ list of words equivalent to the standard name]
            'Duto':         ["tubo", "duto"],
            'Nada':         ["nada"],
            'Confuso':      ["conf", "confuso"],
            'NaoDuto':      ["NaoDuto", "NãoDuto", "Nao-Duto", "Não-Duto", "Nao_Duto", "Não_Duto"],

            'Evento':       ["evnt", "evento"],
            'NaoEvento':    ["nevt", "Nao_Evento"],

            'Anodo':        ["anodo", "anoto"],
            'Flutuadores':  ["boia", "flutuadores"], 
            'Reparo':       ["repr", "reparo"],
            'Dano':         ["dano"],
            'Loop':         ["loop"],
            'Torcao':       ["torc", "torcao"],
            'Gaiola':       ["gaio", "gaiola"],
            'Corrosao':     ["corr", "corrosao"],
            'Enterramento': ["ente", "enterramento"],
            'Cruzamento':   ["cruz", "cruzamento"],
            'Flange':       ["flan", "flange"],

            'Duvida':       ['Duvida', 'Dúvida']
}
no_translation = 'UNTRANSLATED'

net_classes_table = {   # Dict to translate class tags into binary problem tags
                        'rede1':{
                            'Duto':         'Duto',
                            'Nada':         'NaoDuto',
                            'Confuso':      'NaoDuto',
                            'NaoDuto':      'NaoDuto',
                        },
                        'rede2':{   # No translation needed
                            'Evento':       'Evento',
                            'NaoEvento':    'NaoEvento',
                        },
                        'rede3':{   # TODO: This multiclass problem will require a more involved translation
                            'Anodo':        'Anodo',
                            'Flutuadores':  'Flutuadores',
                            'Reparo':       'Reparo',
                            'Dano':         'Dano',
                            'Loop':         'Loop',
                            'Torcao':       'Torcao',
                            'Gaiola':       'Gaiola',
                            'Corrosao':     'Corrosao',
                            'Enterramento': 'Enterramento',
                            'Cruzamento':   'Cruzamento',
                            'Flange':       'Flange',
                            'Duvida':       'Duvida'
                        }
}

# Defines of positive and negative class labels for rede1 and rede2
rede1_positive = "Duto"
rede1_negative = "NaoDuto"
rede2_positive = "Evento"
rede2_negative = "NaoEvento"

# Annotation type names
manual_annotation = "manual"
auto_annotation   = "auto"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
