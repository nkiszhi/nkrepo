#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# read_log.py : Extract scan results of virusshare samples from kav log file. 

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"
__email__ = "zwang@nankai.edu.cn"

import os
import re
import csv

LOG_KAV = "kav.log" # one scan result, raw data 
LOG_RESULT = "result.log" # all scan results, including sample, zip, category, platform, family, result)

def save_csv(list_result):
   
    ### 1. read result.log
    list_all = []
    with open(LOG_RESULT, mode="r", encoding="utf-8") as f:
        list_all = f.read().splitlines()

    ### 2. append new results 
    for x in list_result:
        list_all.append(x)

    ### 3. remove duplicated results 
    list_all = list(set(list_all))
    #list_all = list(filter(lambda x: x, list_all)) # filter empty lines

    ### 4. save results
    with open(LOG_RESULT, "w") as f:
        for r in list_all:
            f.write("{}\n".format(r))


def search_result(line):

    # Search scan results from kav log
    #### 1. Regular expression to match folder and sample
    pattern_vs = r'(VirusShare_[0-9]{5})\\(VirusShare_[a-f0-9]{32})'
    sample_vs = re.search(pattern_vs, line)
    if not sample_vs: # No sample found
        return
    sample = sample_vs.group(2) # vs sample name, for example VirusShare_md5
    folder = sample_vs.group(1) # vs folder, for example "VirusShare_00000" 

    #### 2. Regular expression to match Algorithm:Class.Platform.Famliy.Variant
    pattern_result = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)(\.[a-zA-Z0-9-]*)*'
    result = re.search(pattern_result, line)
    if not result:
        # If no Kaspersky result is found, continue to next kav log
        return

    #### 3. Split result
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
    return (folder, sample, category, platform, family, result)

def read_log(file_log):
    ''' Read Kaspersky log file and extract scan result.'''
    list_result = []
    _n = 0
    #### 1. Read kav log
    print(file_log)
    with open(file_log, mode="r", encoding="utf-8") as f:
        list_result = f.readlines()
    list_result = [x.strip() for x in list_result]
    # File name is VirusShare_MD5 which is more than 43 characters.
    list_result = list(filter(lambda x: len(x) > 100, list_result))

    #### 2. Search sample name and scan results
    list_result = [search_result(x) for x in list_result]
    list_result = list(filter(lambda x: x, list_result)) # filter empty lines
    list_result = list(set(list_result)) # remve duplicated lines

    #### 4. Save result into csv file
    save_csv(list_result)

def main():

    list_result = read_log(LOG_KAV)

if __name__ == "__main__":
    main()
