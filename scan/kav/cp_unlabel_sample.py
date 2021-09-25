#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil
from multiprocessing import Pool

#############################################
# Copy unlabeled samples into a temp folder.
#############################################

HEXSTRING = "0123456789abcdef"
DIR_REPO = "/nkrepo/DATA/sha256/"
DIR_TEMP = "/nkrepo/temp/"

# Copy samples without scan results into temp folder
def cp_samples(folder):
    print(folder)
    files = os.listdir(folder)
    print(len(files))
    for f in files:
        # filter rule 1: only copy samples
        if len(f) != 64:
            continue
        # filter rule 2: only copy samples without  virustotal scan results
        f_json = folder + f + ".json" 
        if os.path.exists(f_json):
            continue
        # filter rule 3: only copy samples without  virustotal scan results
        f_kav = folder + f + ".kav"
        if os.path.exists(f_kav):
            continue
         
        f_sample = folder + f

        shutil.copy(f_sample, DIR_TEMP)
    print("Finished: " + folder)

def main():
    folder_list=[]
    for i in HEXSTRING:
        for j in HEXSTRING:
            for k in HEXSTRING:
                folder_list.append("/nkrepo/DATA/sha256/f/"+i+"/"+j+"/"+k+"/")
    
    p = Pool(50)
    p.map(cp_samples, folder_list)

if __name__=="__main__":
    main()
