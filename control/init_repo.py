#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Init DATA folder with 5-level subfolders.

For example: The hash value of a sample is 0123456xxx. This sample will be saved in the folder "../DATA/0/1/2/3/4/".

"""

import os
import argparse
HEX_STRING = "0123456789abcdef"
LIST_DATA_FOLDER = "list_data_folder.txt"
DIR_SHA256 = ""
DIR_MD5 = ""

# Get the absolute path of DATA folder
dir_data = os.path.abspath("../DATA/")
#print(dir_data)
# Create subfolder list
list_sha256_folder = [ dir_data + "/sha256/" + i + "/" + j + "/" + k + "/" + l + "/" + m + "/" for i in HEX_STRING for j in HEX_STRING for k in HEX_STRING for l in HEX_STRING for m in HEX_STRING]
#list_sha256_folder = [os.path.abspath(x) for x in list_sha256_folder]
list_md5_folder = [ dir_data + "/md5/" + i + "/" + j + "/" + k + "/" + l + "/" + m + "/" for i in HEX_STRING for j in HEX_STRING for k in HEX_STRING for l in HEX_STRING for m in HEX_STRING]
#print(list_sha256_folder[0])
#print(list_md5_folder[0])

# Create subfolders

n = 0
for folder in list_sha256_folder:
    if os.path.exists(folder):
        continue
    os.makedirs(folder)
    n = n + 1
print("Created {} sha256 subfolders".format(n))

n = 0
for folder in list_md5_folder:
    if os.path.exists(folder):
        continue
    os.makedirs(folder)
    n = n + 1
print("Created {} md5 subfolders".format(n))

