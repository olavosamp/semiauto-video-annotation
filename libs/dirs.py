import os

sep     = "/"       # Foward dash for Windows/Linux compatibility
# sep = "\\"        # Backwards dash for Windows specific applications

root                = "."+sep
index               = root+"index"+sep
images              = root+".."+sep+"images"+sep
base_videos         = root+".."+sep+"20170724_FTP83G_Petrobras"+sep
dataset             = root+sep+".."+sep+"datasets"+sep

hashtable           = root+"index"+sep+"localised_video_path_list.csv"
iter_folder         = root+".."+sep+"annotation_loop"+sep
iter_info           = iter_folder+"iter_info.txt"
saved_models        = root+".."+sep+"saved_models"+sep

test                = root+sep+".."+sep+"test"+sep
test_assets         = test+"test_assets"+sep

febe_base_videos    = "/"+"home"+sep+"common"+sep+"flexiveis"+sep+"videos"+sep
febe_images         = root+".."+sep+"images"+sep

def create_folder(path, verbose=False):
    if not(os.path.isdir(path)):
        os.makedirs(path)

create_folder(images)
create_folder(test)
create_folder(saved_models)