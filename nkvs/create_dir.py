#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from multiprocessing import Pool

F_ZIP = "/nkrepo/zip.txt"
DIR_ZIP = "/nkrepo/nkvs/DATA/tmp/"

def worker(zip_name):
    '''Create a folder for each zip file '''
    zip_name = zip_name.replace("\n","").split(".")[0].split("/")[-1]
    # print(zip_name) 
    if os.path.exists(DIR_ZIP + zip_name):
        return
    else:
        print(zip_name)
        os.makedirs(DIR_ZIP + zip_name)

def main():
    zip_list = [] 
    with open(F_ZIP, "r") as f:
        zip_list = f.readlines()
    print(len(list(set(zip_list))))
    p = Pool(20)
    p.map(worker, zip_list)

if __name__=="__main__":
    main()
