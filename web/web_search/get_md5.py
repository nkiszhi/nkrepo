#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import os
import argparse
import pandas as pd
import json

OUTPUT_CSV = "info.csv"
INPUT_CSV = "info.csv"
DIR_SAMPLES = "/home/RaidDisk/nkrepo/web/web_search/samples/"
DIR_MD5 = "../../DATA/md5/"

def create_md5_file():
    # 1. Get all info
    with open(INPUT_CSV, "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    
    # 2. Iterate sample info
    for line in lines:
        list_line = line.split(",")
    # 3. Get sample MD5 and SHA256
        md5 = list_line[0]
        print(md5)
        sha256 = list_line[1]
        print(sha256)
    
    # 4. Get MD5 file path and check existance
        dir_md5 = DIR_MD5 + md5[0] + "/" + md5[1] + "/" + md5[2] + "/" + md5[3] + "/"
        os.makedirs(dir_md5, exist_ok=True)
        f_md5 = dir_md5 + md5
        f_md5 = os.path.abspath(f_md5)
        if os.path.exists(f_md5):
            continue
    # 5. Create MD5 file
        print(f_md5)
        with open(f_md5, "w") as f:
            f.write(sha256)
        break

    return

def add_md5_value():
    with open(INPUT_CSV, "r") as f:
        list_line = f.readlines()
        list_line = [x.rstrip() for x in list_line]

    print(len(list_line))

    list_new_result = []

    for l in list_line:
        list_l = l.split(',') # Get sha256
        sha256 = list_l[0] # Get sha256
        f_json = DIR_SAMPLES + sha256 + ".json" # locate json file
        #print(list_l[0])
        print(f_json)
        if not os.path.exists(f_json):
            print("Json file is not existed")
            continue

        with open(f_json, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                print(key)
            # There are 2 formats of vt json
            if len(data.keys()) == 2:
                md5 = data["results"]["md5"]
            else:
                md5 = data["md5"]

            print(md5)
            #if "scan_date" in data: # locate key "scan_date"
            #    year = data["scan_date"][:4]
            #else:
            #    year = data["results"]["scan_date"][:4] # get year 
            #print(year)

            list_new_result.append(md5 + ',' + l )

    with open(OUTPUT_CSV, "w") as f:
        for l in list_new_result:
            print(l)
            f.write(l+"\n")

def main():
    #add_md5_value()
    create_md5_file()

if __name__ == "__main__":
    main()
