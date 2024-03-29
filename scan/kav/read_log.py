#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# read_log.py : Extract scan results of virusshare samples from kav log file. 
# location: nkrepo/scan/kav/read_log.py

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"
__email__ = "zwang@nankai.edu.cn"

import os
import re
import time
import argparse

FILE_LOG = os.path.abspath("./kav.log")  # Raw log file
#DIR_REPO = os.path.abspath("../../DATA/sha256/")
CSV_RESULT = os.path.abspath("./scan_result.csv")

def save_csv(list_result):
    with open(CSV_RESULT, "w") as f:
        for r in list_result:
            f.write("{}\n".format(r))

#def save_result(kav_result):
#    # Save Kaspersky scan result into kav file
#    #### 1. Split kav result
#    (sha256, algorithm, category, platform, family, result) = kav_result
#    #### 2. Get kav file name. The file name is sha256 value and extension is ".kav"
#    f_kav = DIR_REPO + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256[4] + "/" + sha256 + ".kav"
#    f_kav = os.path.abspath(f_kav)
#    #### 3. Check if kav file is existed. If kav file already existed, return for next kav result.
#    if os.path.exists(f_kav):
#        return 0
#    #### 4. Save kav result into a kav file
#    t = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
#    # Write scan result into kav file
#    with open(f_kav, "w") as f:
#        f.write("{}, {}, {}, {}, {}, {}\n".format(t, result, algorithm, category, platform, family))
#    print("Save scan result into {}.".format(f_kav))
#    return 1
#

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
    ''' Read Kaspersky log file and extract scan result.'''

    list_result = []
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

    #### 3. Save result into kav file
    #_l = [save_result(x) for x in list_result]
    #print("In total, {} kav scan results are saved.".format(sum(_l)))

    #### 4. Save result into csv file
    save_csv(list_result)


def parseargs():
    parser = argparse.ArgumentParser(description="Read and save Kaspersky scan results.")
    parser.add_argument("-l", "--log", help="The Kaspersky log file.", type=str, default=FILE_LOG)
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    file_log = args.log
    if not os.path.exists(file_log):
        print("[X] \"{}\" is not existed.\n".format(file_log))

    read_log(file_log)


if __name__ == "__main__":
    main()
