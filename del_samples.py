#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Add samples into repo."""

import os
import shutil

# folder storing VirusShare.com samples
DIR_SAMPLES = "/home/RaidDisk/nkrepo/samples/"
# folder storing VirusShare.com zip files
DIR_ZIP = "/home/RaidDisk/nkvs/DATA/"
# folder storing all samples at repo
DIR_DATA = "/home/RaidDisk/nkrepo/DATA/"
# password for zip files
PASSWORD = "infected"
list_zip = []

# unzip VirusShare.com zip files in DIR_ZIP and
# store extraced samples to specified folder
# DIR_DATA
def vs_unzip():
    z = os.listdir(DIR_ZIP)
    for f in z:
        path = DIR_ZIP+f
        os.popen("unzip -o -P " + PASSWORD + " " + path + " -d " + DIR_SAMPLES)
        #os.popen("rm -f " + path)
        list_zip.append(path)

# Calculate sample sha256 and move sample to
# repo DATA folder
def vs_sha256():
    files = os.listdir(DIR_SAMPLES)
    for i in files:
        src_path = DIR_SAMPLES + i
        print(src_path)
        if os.path.isdir(src_path):
            continue
        sha256 = os.popen("sha256sum " + src_path)
        sha256 = sha256.read().split(' ')[0]
        #print sha256
        dst_path = DIR_DATA + sha256[0] + "/" +sha256[1] + "/" +sha256[2] + "/" +sha256[3] + "/" + sha256
        #os.popen("mv " + src_path + " " + dst_path)
        shutil.move(src_path, dst_path)
        #print dst_path

def main():
    vs_unzip()
    vs_sha256()

if __name__ == "__main__":
    main()

