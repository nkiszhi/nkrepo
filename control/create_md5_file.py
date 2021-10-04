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
import json
from multiprocessing import Pool
from greet import greet

DIR_SHA256 = "../DATA/sha256/"
DIR_MD5 = "../DATA/md5/"
FILE_LIST_FOLDER = "list_data_folder.txt"
N_WORKER = 2

def get_sha256(f_sample):
    return hashlib.sha256(open(f_sample, "rb").read()).hexdigest()

def get_md5(f_sample):
    return hashlib.md5(open(f_sample, "rb").read()).hexdigest()

def get_md5_file(md5):
    f_md5 = DIR_MD5 + md5[0] + "/" + md5[1] + "/" + md5[2] + "/" + md5[3] + "/" + md5
    return os.path.abspath(f_md5)

def create_md5_file_by_json(f_json):
    # 1. Read json file
    with open(f_json, "r") as f:
        dict_json = json.load(f)

    # 2. Get md5 and sha256 from json
    if len(dict_json.keys()) == 2:
        response_code = dict_json["results"]["response_code"]
        if response_code != 1:
            print("[i] Response code is not 1: {}".format(f_json))
            return 0
        md5 = dict_json["results"]["md5"]
        sha256 = dict_json["results"]["sha256"]
    else:
        response_code = dict_json["response_code"]
        if response_code != 1:
            print("[i] Response code is not 1: {}".format(f_json))
            return 0
        md5 = dict_json["md5"]
        sha256 = dict_json["sha256"]

    # 3. Create md5 file
    f_md5 = get_md5_file(md5)
    if os.path.exists(f_md5):
        return 0
    with open(f_md5, "w") as f:
        f.write("{}\n".format(sha256))
    return 1


def create_md5_file_by_sha256(x):
    f_md5 = x[0]
    sha256 = x[1]

    with open(f_md5, "w") as f:
        f.write("{}\n".format(sha256))
    #print(f_md5)
    return 1


def worker_json(folder):
    _n = 0
    # 1. Get all files in the folder
    list_all = os.listdir(folder)
    _n = len(list_all)
    if not _n:
        return _n

    # 2. Filter no json files
    list_json = list(filter(lambda x:x[64:] == ".json", list_all))
    _n = len(list_json)
    if not _n:
        return _n

    # 3. Get json file path
    list_json = [os.path.abspath(folder + x) for x in list_json]

    # 3. Create md5 files
    #print(folder)
    list_add = list(map(create_md5_file_by_json, list_json))
    return sum(list_add)


def worker_sha256(folder):
    print("[i] {}".format(folder))
    _n = 0
    # 1. Get all files in the folder
    list_all = os.listdir(folder)
    _n = len(list_all)
    if not _n:
        return _n

    # 2. Filter no sha256 files
    list_sha256 = list(filter(lambda x:len(x) == 64, list_all))
    _n = len(list_sha256)
    if not _n:
        return _n
    #print(folder)

    # 3. Get md5 values
    list_sha256_file = [folder + x for x in list_all]
    list_md5 = [get_md5(x) for x in list_sha256_file]

    # 4. Get md5 files
    list_md5_file = [get_md5_file(x) for x in list_md5]
    list_zip = zip(list_md5_file, list_sha256)

    # 5. Filter existed md5 files
    list_zip = list(filter(lambda x: not os.path.exists(x[0]), list_zip))
    if not len(list_zip):
        return 0

    # 6. Create md5 files
    list_add = list(map(create_md5_file_by_sha256, list_zip))
    print("[i] {}: {}".format(folder, sum(list_add)))
    return sum(list_add)

def parseargs():
    parser = argparse.ArgumentParser(description = "Add samples into NKAMG malware repo.")
    parser.add_argument("-t", "--type", help="Create md5 files by sha256 or json", type=str, choices="sha256, json", default="json")
    args = parser.parse_args()
    return args

def main():
    list_folder = [] # List of folders in the malware repo
    _count = [] 
    p = Pool(N_WORKER)

    greet()
    args = parseargs()

    with open(FILE_LIST_FOLDER, "r") as f:
        list_folder = f.readlines()
    list_folder = [os.path.abspath(DIR_SHA256 + x.strip()) + "/" for x in list_folder]
    
    n = 0
    if args.type == "json":
        _count = p.map(worker_json, list_folder)
    elif args.type == "sha256":
        _count = p.map(worker_sha256, list_folder)

    print("{} md5 files are created.".format(sum(_count)))

if __name__ == '__main__':
    main()

