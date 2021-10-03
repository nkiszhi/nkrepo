#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"
__email__ = "zwang@nankai.edu.cn"

import os
import shutil
import argparse
import hashlib
from greet import greet

DIR_SHA256 = "../DATA/sha256/"
DIR_MD5 = "../DATA/md5/"
DIR_TEMP = "./TEMP/"
FILE_LIST_SHA256 = "./list_sha256.txt"
FILE_LIST_MD5 = "./list_md5.txt"

def get_sha256(f_sample):
    return hashlib.sha256(open(f_sample, "rb").read()).hexdigest()

def get_md5(f_sample):
    return hashlib.md5(open(f_sample, "rb").read()).hexdigest()

def add_sample(input_folder):
    # 1. Check folder existence
    if not os.path.exists(input_folder):
        print("[i]: The folder \"{}\" is not exist".format(input_folder)) 
        return

    # 2. Iterate folder
    list_file = os.listdir(input_folder)
    list_file = [os.path.abspath(input_folder + "/" + x) for x in list_file]

    # 3. Get SHA256 and MD5 of samples
    list_sha256 = [get_sha256(x) for x in list_file]
    list_md5 = [get_md5(x) for x in list_file]
    
    # 4. Save SHA256 and MD5 into files
    with open(FILE_LIST_SHA256, "w") as f:
        for i in list_sha256:
            f.write("{}\n".format(i))
    with open(FILE_LIST_MD5, "w") as f:
        for i in list_md5:
            f.write("{}\n".format(i))

    # 5. Copy samples into sha256 repo
    n = 0
    for i in range(len(list_file)):
        #print(i)
        f_src = list_file[i]
        sha256 = list_sha256[i]
        md5 = list_md5[i]
        f_dst = DIR_SHA256 + sha256[0] + "/" +  sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256 
        f_dst = os.path.abspath(f_dst)
        #print(f_dst)
        if os.path.exists(f_dst):
            print("[i] Already existed \"{}\".".format(f_dst))
            continue
        shutil.copy(f_src, f_dst)
        print("[i] Added file \"{}\".".format(f_dst))
        n = n + 1
    # 6. Create md5 file 
        f_md5 = DIR_MD5 + md5[0] + "/" +  md5[1] + "/" + md5[2] + "/" + md5[3] + "/" + md5 
        f_md5 = os.path.abspath(f_md5)
        print("[i] Added file \"{}\".".format(f_md5))
        with open(f_md5, "w") as f:
            f.write(sha256)
        #print(i)
        #print(list_file[i])
        #print(list_sha256[i])
        #print(list_md5[i])

    print("[i] In total, {} samples are added.\n".format(n))
    return n

def parseargs():
    parser = argparse.ArgumentParser(description = "Add samples into NKAMG malware repo.")
    parser.add_argument("-d", "--dir", help="The folder containing samples to add", type=str, default="TEMP")
    args = parser.parse_args()
    return args

def main():
    greet()
    args = parseargs()
    #print(args.dir)
    n = add_sample(args.dir)

if __name__ == '__main__':
    main()

