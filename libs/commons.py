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


# Defines of positive and negative class labels for rede1 and rede2
rede1_positive = "Duto"
rede1_negative = "NaoDuto"
rede2_positive = "Evento"
rede2_negative = "Nao_Evento"

rede3_classes = {0: 'Anodo',
                 1: 'Cruzamento',
                 2: 'Dano',
                 3: 'Flange',
                 4: 'Reparo',
}

net_target_column = {1:'rede1',
                     2:'rede2',
                     3:'rede3'}

# Annotation type names
manual_annotation = "manual"
auto_annotation   = "auto"

net_class_translation_table = {
            # Dict to normalize class annotations into standard class tags
            # This translation table follows the non-standard format described below.
            # Standard class name:  [ list of words equivalent to the standard name]
            'Duto':             ["tubo", "duto", "duct"],
            'Nada':             ["nada", "not-duct", "notduct", "not_duct"],
            'Confuso':          ["conf", "confuso", "confusion"],
            'NaoDuto':          ["NaoDuto", "NãoDuto", "Nao-Duto", "Não-Duto", "Nao_Duto", "Não_Duto"],

            'Evento':           ["evnt", "evento"],
            'NaoEvento':        ["nevt", "Nao_Evento"],

            'Anodo':            ["anodo", "anoto", "anôdo"],
            'NaoAnodo':         ["NaoAnodo"],
            'Flutuadores':      ["boia", "flutuadores", "flut"], 
            'NaoFlutuadores':   ["NaoFlutuadores"],
            'Reparo':           ["repr", "reparo"],
            'NaoReparo':        ["NaoReparo"],
            'Dano':             ["dano"],
            'NaoDano':          ["NaoDano"],
            'Loop':             ["loop"],
            'NaoLoop':          ["NaoLoop"],
            'Torcao':           ["torc", "torcao"],
            'NaoTorcao':        ["NaoTorcao"],
            'Gaiola':           ["gaio", "gaiola"],
            'NaoGaiola':        ["NaoGaiola"],
            'Corrosao':         ["corr", "corrosao"],
            'NaoCorrosao':      ["NaoCorrosao"],
            'Enterramento':     ["ente", "enterramento"],
            'NaoEnterramento':  ["NaoEnterramento"],
            'Cruzamento':       ["cruz", "cruzamento"],
            'NaoCruzamento':    ["NaoCruzamento"],
            'Flange':           ["flan", "flange"],
            'NaoFlange':        ["NaoFlange"],

            'Duvida':       ['Duvida', 'Dúvida'],
}
no_translation = 'UNTRANSLATED'

net_binary_table = { # Dict to translate standard class tags into binary problem tags
                     # Values should be only keys of net_class_translation_table
                        'rede1':{
                            'Duto':         'Duto',
                            'Nada':         'NaoDuto',

                            'Confuso':      'NaoDuto',
                            'NaoDuto':      'NaoDuto',
                        },
                        'rede2':{ # No translation needed
                            'Evento':       'Evento',
                            'NaoEvento':    'NaoEvento',
                            'Duvida':       'Duvida',
                        },
                        'rede3':{
                            'Anodo':          rede3_classes[0],
                            'NaoAnodo':       'NaoAnodo',
                            'anod':           rede3_classes[0],
                            'anoto':          rede3_classes[0],

                            'Flutuadores':    'Flutuadores',
                            'NaoFlutuadores': 'NaoFlutuadores',
                            'flut':           'Flutuadores',

                            'Reparo':         rede3_classes[4],
                            'NaoReparo':      'NaoReparo',
                            'repr':           rede3_classes[4],

                            'Dano':           rede3_classes[2],
                            'NaoDano':        'NaoDano',

                            'Loop':           'Loop',
                            'NaoLoop':        'NaoLoop',

                            'Torcao':         'Torcao',
                            'NaoTorcao':      'NaoTorcao',
                            'torc':           'Torcao',

                            'Gaiola':         'Gaiola',
                            'NaoGaiola':      'NaoGaiola',
                            'gaio':           'Gaiola',

                            'Corrosao':       'Corrosao',
                            'NaoCorrosao':    'NaoCorrosao',
                            'Corrosão':       'Corrosao',
                            'corr':           'Corrosao',

                            'Enterramento':   'Enterramento',
                            'NaoEnterramento':'NaoEnterramento',
                            'ente':           'Enterramento',

                            'Cruzamento':     rede3_classes[1],
                            'NaoCruzamento':  'NaoCruzamento',
                            'cruz':           rede3_classes[1],

                            'Flange':         rede3_classes[3],
                            'NaoFlange':      'NaoFlange',
                            'flan':           rede3_classes[3],

                            'Duvida':         'Duvida'
                        }
}

# Priority table for multilabel conflict desambiguation
# rede_3_multiclass_priority_table = [rede3_classes[2], # Dano
#                                     rede3_classes[4], # Reparo
#                                     rede3_classes[0], # Anodo
#                                     rede3_classes[1], # Cruzamento
#                                     rede3_classes[3], # Flange
# ]
# Temporary priority table to test fusion function
rede_3_multiclass_priority_table = [rede3_classes[2], # Dano
                                    rede3_classes[0], # Anodo
                                    rede3_classes[1], # Cruzamento
                                    rede3_classes[3], # Flange
                                    rede3_classes[4], # Reparo
]

## Defines
IMAGENET_MEAN       = [0.485, 0.456, 0.406]
IMAGENET_STD        = [0.229, 0.224, 0.225]
MPL_FIG_SIZE        = (18, 9)
MPL_FIG_SIZE_SMALL  = (8, 4)

FRAME_HASH_COL_NAME = 'FrameHash'

## Ad-hoc information
# Roberto dataset info
val_videos_reference_dataset_rede_1 =[ # 3-ple of (report, DVD number, video name)
                                        ("GHmls16-263_OK", "1", "20161101212059250@DVR-SPARE_Ch1.wmv"),
                                        ("CIMRL10-676_OK", None, "PIDF-1 PO MRL-021_parte1.mpg"),
                                        ("CIMRL10-676_OK", None, "PIDF-1 PO MRL-021_parte2.mpg"),
                                        ("TCOpm16-140_OK", "6", "VTS_01_1.VOB"),
                                        ("TCOpm16-140_OK", "2", "VTS_01_2.VOB"),
                                        ("TCOpm16-140_OK", "2", "VTS_01_3.VOB"),
                                        ("TCOpm16-140_OK", "2", "VTS_01_4.VOB"),
                                        ("TCOpm16-140_OK", "2", "VTS_01_5.VOB"),
]
val_videos_reference_dataset_rede_1_hashes = [
                                        'c2ceb075adf095c97fe8944f7aa321c2',
                                        '304248734049c5da816e12c3bcecbb76',
                                        'b40c46629f8630e1ccfc57ddf105e1f8',
                                        '02307d50a3d2c94a2427083bb0491bff',
                                        'ddd7dc17eee8665db5be7fa324177c37',
                                        '37c0ab0aef7840dcd33a4e43494c413f',
                                        '58717dc038f96fca74e5939f85db509f',
                                        '6a6fcd20e5728b2a129f6e5cfcf39d5e',
]