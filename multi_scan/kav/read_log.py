#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# read_log.py : Extract kav scan results of samples from log file. 
# location: nkrepo/scan/kav/read_log.py

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2024 NKAMG"
__license__ = "GPL"
__email__ = "zwang@nankai.edu.cn"

import os
import re
import time
import argparse
import csv

list_result = []
file_result = ""

def save_csv(): 
    n = len(list_result)
    with open(file_result, 'w', newline='', encoding='utf-8') as file_csv:  
       writer = csv.writer(file_csv)  
       for row in list_result:
          writer.writerow(row)  
    print("[o] Save {} kav results into {} file.".format(n, file_result))


def search_result(line):
    # Search scan results from kav log
    #### 1. Regular expression for sha256
    #pattern_sha256 = r'[a-f0-9]{64}'
    pattern_vs = r'VirusShare_[a-f0-9]{32}'
    #### 2. Regular expression to match Algorithm:Class.Platform.Famliy.Variant
    pattern_result = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)(\.[a-zA-Z0-9-]*)*'
    #### 3. Search sha256
    #sha256 = re.search(pattern_sha256, line)
    #if not sha256:
    #    # If no sha256 string found, continue to next kav log
    #    return
    #sha256 = sha256.group()
    file_vs = re.search(pattern_vs, line)
    if not file_vs:
        return
    file_vs = file_vs.group()

    #### 4. Search scan result
    result = re.search(pattern_result, line)
    if not result:
        # If no Kaspersky result is found, continue to next kav log
        return
    #### 5. Split result
    if result.group(1):
        algorithm = result.group(1)
    else:
        algorithm = ""
    # category, such as Trojan, Worm,...
    category = result.group(2)
    # malware running platform, such as Win32, Script, ...
    platform = result.group(3)
    # malware family
    family = result.group(4)
    # malware variant information
    mal_variant = result.group(5)
    # detection result: Algorithm:Class.Platform.Family.Variant
    result = result.group()
    return (file_vs, category, platform, family, result)

def read_log(file_log):
    global list_result
    ''' Read Kaspersky log file and extract scan result.'''
    _n = 0
    #### 1. Read kav log
    print(file_log)
    with open(file_log, mode="r", encoding="utf-8") as f:
        list_result = f.readlines()
    list_result = [x.strip() for x in list_result]
    # File name is VirusShare_MD5 which is more than 43 characters.
    list_result = list(filter(lambda x: len(x) > 50, list_result))

    #### 2. Search sample name and scan results
    list_result = [search_result(x) for x in list_result]
    list_result = list(filter(lambda x: x, list_result)) # filter empty lines
    list_result = list(set(list_result)) # remve duplicated lines

    n = len(list_result)
    if n:
        print("[o] Extract {} kav scan results from {} file.".format(n, file_log))
        return True
    else:
        print("[!] No kav scan results extracted from {} file.".format(file_log))
        return False


def parseargs():
    parser = argparse.ArgumentParser(description="Read and save Kaspersky scan results.")
    parser.add_argument("-l", "--log", help="The Kaspersky log file.", type=str, default="kav.log")
    parser.add_argument("-r", "--result", help="The scan result file in csv format.", type=str, default="result.csv")
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    global file_result
    file_log = args.log
    file_result=args.result
    if not os.path.exists(file_log):
        print("[!] \"{}\" is not existed.\n".format(file_log))
        return False

    # Extract kav scan result
    if not read_log(file_log):
        return False

    # Save result to csv file
    save_csv()


if __name__ == "__main__":
    main()
