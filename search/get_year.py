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

def main():
    with open("scan_result.csv", "r") as f:
        list_line = f.readlines()

    print(len(list_line))

    list_new_result = []

    for l in list_line:
        list_l = l.strip().split(',') # Get sha256
        sha256 = list_l[0] # Get sha256
        f_json = "/home/RaidDisk/new/nkrepo/search/samples/{}.json".format(sha256) # locate json file
        #print(list_l[0])
        print(f_json)
        if not os.path.exists(f_json):
            print("Json file is not existed")
            continue
        with open(f_json, "r") as f:
            data = json.load(f)
            #for key, value in data.items():
            #    print(key)
            if "scan_date" in data: # locate key "scan_date"
                year = data["scan_date"][:4]
            else:
                year = data["results"]["scan_date"][:4] # get year 
            print(year)

            list_new_result.append(l.strip() + ", " + year )

    with open(OUTPUT_CSV, "w") as f:
        for l in list_new_result:
            print(l)
            f.write(l+"\n")

if __name__ == "__main__":
    main()
