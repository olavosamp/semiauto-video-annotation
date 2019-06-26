import os

sep     = "/"       # Foward dash for Windows/Linux compatibility
# sep = "\\"        # Backwards dash for Windows specific applications

root                = "."+sep
index               = root+"index"+sep+"main_index.csv"
images              = root+".."+sep+"images"+sep
base_videos         = root+".."+sep+"20170724_FTP83G_Petrobras"+sep
dataset             = root+sep+".."+sep+"datasets"+sep

febe_base_videos    = "/"+"home"+sep+"common"+sep+"flexiveis"+sep+"videos"+sep
febe_images         = root+sep+".."+sep+"images"+sep

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
