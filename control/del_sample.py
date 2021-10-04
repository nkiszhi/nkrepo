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

FILE_LIST_SHA256 = "list_sha256.txt"

DIR_MD5 = "../DATA/md5/"
DIR_SHA256 = "../DATA/sha256/"

list_samples = []

def get_md5(file_name):
    return hashlib.md5(open(file_name, "rb").read()).hexdigest()

def get_abspath_sha256(sha256):
    return os.path.abspath("{}/{}/{}/{}/{}/{}".format(DIR_SHA256, sha256[0], sha256[1], sha256[2], sha256[3], sha256))

def get_abspath_md5(md5):
    return os.path.abspath("{}/{}/{}/{}/{}/{}".format(DIR_MD5, md5[0], md5[1], md5[2], md5[3], md5))

def del_file(file_name):
    #print(file_name)
    if os.path.exists(file_name):
        os.remove(file_name)
        print("[i] Delete \"{}\"".format(file_name))

def del_sample(file_sha256):
    list_sha256 = []
    with open(file_sha256, "r") as f:
        list_sha256 = f.readlines()
    list_sha256 = [x.strip() for x in list_sha256]
    list_file_sha256 = [get_abspath_sha256(x) for x in list_sha256]
    list_file_sha256 = list(filter(lambda x: os.path.exists(x), list_file_sha256))
    n = len(list_file_sha256)
    if not n: 
        print("[i] No samples need deleting.")
        return
    list_file_md5 = [get_abspath_md5(get_md5(x)) for x in list_file_sha256]
    #print(list_file_sha256[0])
    #print(list_file_md5[0])
    #print(len(list_file_md5))
    #print(len(list_file_sha256))
    for i in range(n):
        del_file(list_file_sha256[i])
        del_file(list_file_md5[i])
    print("[i] {} samples are deleted.\n".format(n))
    return n

def parseargs():
    parser = argparse.ArgumentParser(description = "Delete samples in the repository by sha256.")
    parser.add_argument("-f", 
        "--file", 
        help="input a file containing a list of sha256 values", 
        type=str, 
        default=FILE_LIST_SHA256)
    args = parser.parse_args()
    return args

def main():
    greet()
    args = parseargs()
    n = del_sample(args.file)


if __name__ == '__main__':
    main()
