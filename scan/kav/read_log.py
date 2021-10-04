#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"
__email__ = "zwang@nankai.edu.cn"

import os
import re
import time
import argparse
from greet import greet

KAV_LOG = "kav.log" # Raw log file
DIR_REPO = "../../DATA/sha256/"
CSV_RESULT = "./scan_result.csv"

def save_csv(list_scan_result):
    with open(CSV_RESULT, "w") as f:
        for r in list_scan_result:
            f.write("{}\n".format(r))


def save_result(kav_result):
    # Save Kaspersky scan result into kav file
    (sha256, algorithm, category, platform, family, result) = kav_result
    # The file name is sha256 value and extension is ".kav"
    f_kav = DIR_REPO + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256 + ".kav"
    #if os.path.exists(f_kav):
    #    return 0
    t = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # Write scan result into kav file
    with open(f_kav, "w") as f:
        f.write("{}, {}, {}, {}, {}, {}\n".format(t, result, algorithm, category, platform, family))
    print(f_kav)
    return 1

def search_result(line):
    # Search kav scan results from kav log
    #### 1. Regular expression for sha256
    pattern_sha256 = r'[a-f0-9]{64}'
    #### 2. Regular expression to match Algorithm:Class.Platform.Famliy.Variant
    pattern_result = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)(\.[a-zA-Z0-9-]*)*'
    #### 3. Search sha256
    sha256 = re.search(pattern_sha256, line)
    if not sha256:
        # If no sha256 string found, continue to next kav log
        return
    sha256 =  sha256.group()
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
    #mal_variant = result.group(5)
    # detection result: Algorithm:Class.Platform.Family.Variant
    result = result.group()
    return (sha256, algorithm, category, platform, family, result)

def read_log(f_kav_log):
    ''' Read Kaspersky raw scan log file and extract samples detection information.
The extracted information is stored in kav_results.txt already.'''

    list_scan_result = [] 
    scan_list = []
    _n = 0
    # 1. Read results from kav log
    print(f_kav_log)
    with open(f_kav_log, mode = "r", encoding = "utf-8") as f:
        list_result = f.readlines()
    list_result = [x.strip() for x in list_result]
    list_result = list(filter(lambda x: len(x) > 80, list_result))

    # 2. Search SHA256 and scan results
    list_result = [search_result(x) for x in list_result]
    list_result = list(filter(lambda x: x, list_result))
    list_result = list(set(list_result))
    
    # 3. Save result into kav file 
    _l = [save_result(x) for x in list_result]
    print(sum(_l))

    # 4. Save result into csv file
    #save_csv(list_result)

def parseargs():
    parser = argparse.ArgumentParser(desription = "Read and save Kaspersky scan results.")
    parser.add_argument("-l", "--log", help="The Kaspersky log file.", type=str, default=KAV_LOG)
    args = parser.parse_args()
    return args
 
def main():
    greet()
    args = parseargs()
    f_kav_log = args.log
    if not os.path.exists(f_kav_log):
        print("[i] The\"{}\" is not existed.\n".format(f_kav_log))

    read_log(f_kav_log)

    
if __name__ == "__main__":
    main()
