#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# folder for virusshare samples
DIR_DATA = "/home/RaidDisk/"
# folder for virusshare zip files
DIR_ZIP = "/home/RaidDisk/virusshare/"
# password for zip files
PASSWORD = "infected"

# unzip virusshare zip files in DIR_ZIP to specified folder DIR_DATA
def vs_unzip():
    z = os.listdir(DIR_ZIP)
    for f in z:
        path = DIR_ZIP+f
        os.popen("unzip -o -P " + PASSWORD + " " + path + " -d " + DIR_DATA)
        os.popen("rm -f " + path)
    pass

# calculate sample sha256 and mv sample to specified folder
def vs_sha256():
    files = os.listdir(DIR_DATA)
    for i in files:
        path = DIR_DATA + i
        print path
        if os.path.isdir(path):
            continue
        
        sha256 = os.popen("sha256sum " + path)
        sha256 = sha256.read().split(' ')[0]
        #print sha256
        dst_path = DIR_DATA + sha256[0] + "/" +sha256[1] + "/" +sha256[2] + "/" +sha256[3] + "/" + sha256
        os.popen("mv " + path + " " + dst_path)
        print dst_path

    pass

def main():
    vs_unzip()
    vs_sha256()

main()
