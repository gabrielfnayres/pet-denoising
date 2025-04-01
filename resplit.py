import os
import sys
import shutil
from copy import deepcopy
from sympy import solve_triangulated


PATH_TRAIN = "/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/dataset/train_mat"
PATH_TEST = "/Users/fnayres/upenn/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/dataset/test_mat"

def resplit() -> None:
    
    # SPLITING FILE NAMES
    files_train = os.listdir(PATH_TRAIN)
    files_test = os.listdir(PATH_TEST)


    s_train = len(files_train)
    s_test = len(files_test)

    new_train = []
    for file in files_train:
        
        if len(new_train) < s_train/2:
            new_train.append(file)

    new_test = []
    for file in files_test:
        if len(new_test) < s_test/2:
            new_test.append(file)

    #RECREATE THE DIRS
    if new_test is not None and new_train is not None:
        os.makedirs("./halfdataset")
        os.makedirs("./halfdataset/train")
        os.makedirs("./halfdataset/test")
        
        for i in range(0,len(new_train)//2):
            shutil.copyfile(PATH_TRAIN+"/"+new_train[i],"./halfdataset/train/"+new_train[i])    
        
        for i in range(0, len(new_test)//2):
            shutil.copyfile(PATH_TEST+"/"+new_test[i],"./halfdataset/test/"+new_test[i])
    

resplit()


    

