import numpy                as np
import pandas               as pd
from pathlib                import Path
from glob                   import glob
from tqdm                   import tqdm

import libs.dirs            as dirs
import libs.commons         as commons
import libs.dataset_utils   as dutils

def get_input_target_class(net_class_list):
    classLen = len(net_class_list)
    print("Enter the target class code from list:\n")
    print("Code\tClass name")
    for i in range(classLen):
        print("{}:\t{}".format(i, net_class_list[i]))

    input_class_code = int(input())
    if input_class_code < classLen:
        event_class = net_class_list[i]
    else:
        event_class = "UNKNOWN"

    while event_class not in net_class_list:
        input_class_code = int(input("Unknown class. Please select a class from the list."))
        
        if input_class_code < classLen:
            event_class = net_class_list[i]
    return event_class

# Get inputs
rede        = int(input("\nEnter net number.\n"))

datasetName = "full_dataset_rede_{}".format(rede)
loopFolder        = Path(dirs.iter_folder) / datasetName
prevAnnotatedPath = loopFolder / "iteration_0/final_annotated_images_rede_{}.csv".format(rede-1)

if rede == 2:
    baseClass = commons.rede1_positive
elif rede == 3:
    baseClass = commons.rede2_positive
    event_class = get_input_target_class(commons.rede3_classes)
else:
    raise NotImplementedError("Only implemented for rede2 and rede3.")

print(event_class)
exit()
# Create new labeled and unlabeled datasets from previous level annotated images
dutils.start_loop(prevAnnotatedPath, baseClass, commons.net_target_column[rede-1])
