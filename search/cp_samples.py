#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil

#############################################
# Copy unlabeled samples into a temp folder.
#############################################

DIR_SAVE = "/home/RaidDisk/new/nkrepo/search/samples/"
FILE_SHA256 = "sha256.txt"
DIR_REPO = "/home/RaidDisk/nkrepo/DATA/"
FILE_SCAN_RESULT = "scan_result.csv"

def copy_samples(f_scan_result, d_save):
    _n = 0
    list_sha256 = []

    with open(f_scan_result, "r") as f:
        list_scan_result = f.readlines()

    _n = 0
    for r in list_scan_result:
        sha256 = r.split(",")[0]
        src_file = DIR_REPO + sha256[0] + '/' + sha256[1] + '/' + sha256[2] + '/' + sha256[3] + '/' + sha256
        if not os.path.exists(src_file):
            continue
        src_json_file = src_file + ".json" 
        if not os.path.exists(src_json_file):
            continue
        #print(src_file)
        #print(src_json_file)
        dst_file = DIR_SAVE + sha256
        dst_json_file = DIR_SAVE + sha256 + ".json"
        #print(dst_file)
        #print(dst_json_file)
        _n = _n + 1
        print("{}: {}".format(_n, sha256))
        shutil.copyfile(src_file, dst_file)
        shutil.copyfile(src_json_file, dst_json_file)
            
def parse_args():
    parser = argparse.ArgumentParser(description = "Search samples.")
    parser.add_argument("-f", "--file_scan_result", help="The file containing Kaspersky scan results.", type=str, default=FILE_SCAN_RESULT)
    parser.add_argument("-d", "--dst_dir", help="The directory to store the samples.", type=str, default=DIR_SAVE)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    copy_samples(args.file_scan_result, args.dst_dir)

if __name__=="__main__":
    main()
