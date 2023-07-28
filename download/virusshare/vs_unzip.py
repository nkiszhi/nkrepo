#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vs_unzip.py:  Extract files from zip file and save the files into specific folder
# location: nkrepo/download/nkvs/vs_unzip.py

import os
from multiprocessing import Pool

ZIP_FILE = os.path.abspath("./DATA/zip.txt") # nkrepo/download/nkvs/DATA/zip.txt
VS_DATA = os.path.abspath("./DATA/")


def worker(zip_name):
    zip_folder = VS_DATA + "/" + zip_name.split(".")[0].split("/")[-1] # remove ".zip" and previous path
    if not os.path.exists(zip_folder):
        os.makedirs(zip_folder)
    # unzip -P infected .zip -d /nkrepo/nkvs/DATA/tmp/()
    os.system("unzip -P infected " + zip_name + " -d " + zip_folder)

def main():

    list_zip = []
    with open(ZIP_FILE, "r") as f:
        list_zip = f.readlines()
    list_zip = [x.strip() for x in list_zip]
    p=Pool(50)
    p.map(worker, list_zip)

if __name__=="__main__":
    main()






