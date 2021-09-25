#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil
from multiprocessing import Pool
import time

#############################################
# Copy unlabeled samples into a temp folder.
#############################################

HEXSTRING = "0123456789abcdef"
DIR_REPO = "/nkrepo/DATA/sha256/"

# Copy samples without scan results into temp folder
def add_time(folder):
    print(folder)
    files = os.listdir(folder)
    print(len(files))
    t = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    folder_list=[]
    for f in files:
        # filter rule 1: only copy samples
        if len(f) != 68:
            continue
        if f[-4:] != ".kav":
            continue
        print(f)
        f_kav = folder + f
        f = open(f_kav, "r")
        l = f.readline()
        print(l)
        f.close()
        l = t + ", " + l 
        f = open(f_kav, "w")
        f.write(l)
        f.close()

    print("Finished: " + folder)

def main():
    for i in HEXSTRING:
        for j in HEXSTRING:
            for k in HEXSTRING:
                for l in HEXSTRING:
                    folder = DIR_REPO + i + "/" + j + "/" + k + "/" + l + "/"
                    add_time(folder)
    

if __name__=="__main__":
    main()
