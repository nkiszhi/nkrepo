#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"
__email__ = "zwang@nankai.edu.cn"

"""Count sha256, md5, vt and kav files in the repo."""

import os
import argparse
from greet import greet
from multiprocessing import Pool

DIR_SHA256 = "../DATA/sha256/" # The repo storing json files
DIR_MD5 = "../DATA/md5/" # The repo storing json files
LIST_DATA_FOLDER = "list_data_folder.txt"
N_WORKER = 200

def worker_sha256(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if len(f) == 64: # The length of SHA256 file name is 64 bytes
            #print(f)
            _n += 1
    return _n

def worker_md5(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if len(f) == 32: # The length of MD5 file name is 32 bytes.
            #print(f)
            _n += 1
    return _n

def worker_kav(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if f[64:] == ".kav": # The file name of Kaspersky scan result uses '.kav' as extension
            #print(f)
            _n += 1
    return _n

def worker_vt(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if f[64:] == ".json": # The file name of VirusTotal scan result uses '.json' as extension.
            #print(f)
            _n += 1
    return _n

def worker_nolabel(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if len(f) == 64:
            f_json = folder + "/" + f + ".json"
            f_kav = folder + "/" + f + ".kav"
            if os.path.exists(f_json): # Have virustotal result
                continue
            if os.path.exists(f_kav): # Have Kaspersky result
                continue
            #print(f)
            _n += 1
    return _n

# Count sha256 samples
def count_sha256(list_folder):
    p = Pool(N_WORKER)
    list_folder = [os.path.abspath(DIR_SHA256 + x) for x in list_folder]
    _count = []
    _count = p.map(worker_sha256, list_folder)
    print("There are {} samples in the repo.".format(sum(_count)))

# Count md5 samples
def count_md5(list_folder):
    p = Pool(N_WORKER)
    list_folder = [os.path.abspath(DIR_MD5 + x) for x in list_folder]
    _count = []
    _count = p.map(worker_md5, list_folder)
    print("There are {} samples with MD5 file in the repo.".format(sum(_count)))

# Count VirusTotal scan results
def count_vt(list_folder):
    p = Pool(N_WORKER)
    list_folder = [os.path.abspath(DIR_SHA256 + x) for x in list_folder]
    _count = []
    _count = p.map(worker_vt, list_folder)
    print("There are {} samples with VirusTotal scan result in the repo.".format(sum(_count)))

# Count Kaspersky scan results
def count_kav(list_folder):
    p = Pool(N_WORKER)
    list_folder = [os.path.abspath(DIR_SHA256 + x) for x in list_folder]
    _count = []
    _count = p.map(worker_kav, list_folder)
    print("There are {} samples with Kaspersky scan result in the repo.".format(sum(_count)))

# Count samples without Kaspersky scan result nor VirusTotal
# result
def count_nolabel(list_folder):
    p = Pool(N_WORKER)
    list_folder = [os.path.abspath(DIR_SHA256 + x) for x in list_folder]
    _count = []
    _count = p.map(worker_nolabel, list_folder)
    print("There are {} samples with Kaspersky scan result in the repo.".format(sum(_count)))

def parse_args():
    parser = argparse.ArgumentParser(description = "Count samples in the repo.", 
             epilog = "NKAMG")
    parser.add_argument("-t", 
        "--target", 
        help="Count target files in the repo, including sha256, md5, vt, kav, nolabel.", 
        type=str, 
        choices = ["sha256", "md5", "vt", "kav", "nolabel"], 
        default="sha256")
    args = parser.parse_args()
    return args

def main():
    greet()
    args = parse_args()

    with open(LIST_DATA_FOLDER, "r") as f:
        list_folder = f.readlines()
    list_folder = [x.strip() for x in list_folder]

    if args.target == "sha256":
        count_sha256(list_folder)
    elif args.target == "md5":
        count_md5(list_folder)
    elif args.target == "vt":
        count_vt(list_folder)
    elif args.target == "kav":
        count_kav(list_folder)
    elif args.target == "nolabel":
        count_nolabel(list_folder)

if __name__ == "__main__":
    main()
