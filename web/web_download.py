#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import os
import shutil
import tarfile
from torrentool.api import Torrent


DIR_DATA = "../DATA/sha256/"
DIR_DOWNLOAD = "./DOWNLOAD/"
DIR_DOWNLOAD_TORRENT = "./DOWNLOAD/TORRENT/" # a folder to store samples for p2p download
DIR_DOWNLOAD_TGZ = "./DOWNLOAD/TGZ/" # a folder to store samples for tgz download
DIR_TEMP = "./DOWNLOAD/TEMP/"


# This function is used for Web 
def get_torrent_file(sha256):
    #### 1. Locate the sample for downloading
    f_src = DIR_DATA + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256
    f_src = os.path.abspath(f_src)
    print("[p2p] Download sample: {}".format(f_src))

    #### 2. Create a folder to store sample 
    dir_download = DIR_DOWNLOAD_TORRENT + sha256 + "/"
    dir_download = os.path.abspath(dir_download) + "/"
    os.makedirs(dir_download, exist_ok=True)
    print("[p2p] Download folder: {}".format(dir_download))
    
    #### 3. Copy the sample into download folder
    f_dst = dir_download + sha256
    f_dst = os.path.abspath(f_dst)
    shutil.copyfile(f_src, f_dst)
    print("[p2p] Copy sample to {}".format(f_dst))

    #### 4. Generate torrent file
    new_torrent = Torrent.create_from(dir_download)
    f_torrent = DIR_DOWNLOAD_TORRENT + sha256 + ".torrent" 
    f_torrent = os.path.abspath(f_torrent)
    new_torrent.to_file(f_torrent)
    print("[p2p] Create torrent file {}".format(f_torrent))

    #### 5. Return torrent file
    return f_torrent

# This function is used for Web 
def get_tgz_file(sha256):
    #### 1. Locate the sample for downloading
    f_src = DIR_DATA + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256
    f_src = os.path.abspath(f_src)
    print("[tgz] Download sample: {}".format(f_src))

    #### 2. Create a folder to store sample 
    dir_download = DIR_DOWNLOAD_TGZ + sha256 + "/"
    dir_download = os.path.abspath(dir_download) + "/"
    os.makedirs(dir_download, exist_ok=True)
    print("[tgz] Download folder: {}".format(dir_download))
    
    #### 3. Copy the sample into download folder
    f_dst = dir_download + sha256
    f_dst = os.path.abspath(f_dst)
    shutil.copyfile(f_src, f_dst)
    print("[tgz] Copy sample to {}".format(f_dst))

    #### 4. Generate tgz file
    f_tgz = DIR_DOWNLOAD_TGZ + sha256 + ".tgz"
    f_tgz = os.path.abspath(f_tgz)
    with tarfile.open(f_tgz, "w:gz") as tar:
        tar.add(f_src, arcname=os.path.basename(f_src))
    #shutil.rmtree(dir_download) # remove the temp folder
    print("[tgz] Create tgz file {}".format(f_tgz))

    #### 5. Return torrent file
    return f_tgz

def del_dir_files():
    for i in os.listdir(DIR_TEMP):
        f_del = DIR_TEMP + i
        f_del = os.path.abspath(d_del)
        os.remove(f_del)

def get_tgz_files(list_sha256):

    if not list_sha256:
        return ""
    #### 1. Clean TEMP folder
        del_dir_files()

    #### 2. Mov files into TEMP folder
    for sha256 in list_sha256:
        # 1.1 locate source file
        f_src = DIR_DATA + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256
        f_src = os.path.abspath(f_src)
        if not os.path.exists(f_src):
            continue
        # 1.2 locate destination file
        f_dst = DIR_TEMP + sha256
        f_dst = os.path.abspath(f_dst)

        # 1.3 copy samples into TEMP folder
        shutil.copy(f_src, f_dst)

    #### 3. Generate tgz file
    f_tgz = DIR_DOWNLOAD + "list.tgz"
    f_tgz = os.path.abspath(f_tgz)
    with tarfile.open(f_tgz, "w:gz") as tar:
        tar.add(DIR_TEMP, arcname=os.path.basename(DIR_TEMP))
    #shutil.rmtree(dir_download) # remove the temp folder
    print("[tgz] Create tgz file {}".format(f_tgz))

    #### 5. Return torrent file
    return f_tgz

    

def get_torrent_files(list_sha256):

    if not list_sha256:
        return ""
    #### 1. Clean TEMP folder
        del_dir_files()

    #### 2. Mov files into TEMP folder
    for sha256 in list_sha256:
        # 1.1 locate source file
        f_src = DIR_DATA + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256
        f_src = os.path.abspath(f_src)
        if not os.path.exists(f_src):
            continue
        # 1.2 locate destination file
        f_dst = DIR_TEMP + sha256
        f_dst = os.path.abspath(f_dst)

        # 1.3 copy samples into TEMP folder
        shutil.copy(f_src, f_dst)

    #### 3. Generate torrent file
    new_torrent = Torrent.create_from(DIR_TEMP)
    f_torrent = DIR_DOWNLOAD + "list.torrent" 
    f_torrent = os.path.abspath(f_torrent)
    new_torrent.to_file(f_torrent)
    print("[p2p] Create torrent file {}".format(f_torrent))

    #### 4. Return torrent file
    return f_torrent


def main():
    print("Provide functions for web downloading")
    

if __name__ == "__main__":
    main()
