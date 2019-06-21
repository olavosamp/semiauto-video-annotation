import os

sep     = "/"       # Foward dash for Windows/Linux compatibility
# sep = "\\"        # Backwards dash for Windows specific applications

root                = "."+sep
csv                 = root+"csv"+sep
images              = root+".."+sep+"images"+sep
base_videos         = root+".."+sep+"20170724_FTP83G_Petrobras"+sep
index               = root+"csv"+sep+"main_index.csv"
dataset             = root+sep+".."+sep+"datasets"+sep

def create_folder(path, verbose=False):
    try:
        os.makedirs(path)
    except OSError as e:
        # Folder already exists or destFolder is invalid
        if verbose:
            print(e)
        else:
            pass

create_folder(images)
