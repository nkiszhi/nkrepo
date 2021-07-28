#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool

ZIP_FILE = "/nkrepo/zip.txt"
ZIP_DIR = "/nkrepo/nkvs/DATA/tmp/"


def worker(zip_name):
    ''' Extract files from zip file and save the files into specific folder ''' 
    zip_name = zip_name.replace("\n","")
    zip_folder = ZIP_DIR + zip_name.split(".")[0].split("/")[-1] # remove ".json" and previous path
    # unzip -P infected .zip -d /nkrepo/nkvs/DATA/tmp/()
    os.system("unzip -P infected " + zip_name + " -d " + zip_folder)

def main():

    zip_list=[]
    with open("/nkrepo/zip.txt","r") as f:
        zip_list = f.readlines()

    p=Pool(20)
    p.map(worker, zip_list)

if __name__=="__main__":
    main()
