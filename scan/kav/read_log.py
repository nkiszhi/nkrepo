#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time

KAV_LOG = "kav.log" # Raw log file
DIR_REPO = "/nkrepo/DATA/sha256/"
CSV_RESULT = "./scan_result.csv"

def save_csv(list_scan_result):
    with open(CSV_RESULT, "w") as f:
        for r in list_scan_result:
            f.write("{}\n".format(r))


def save_result(sha256, result, algorithm, mal_class, mal_platform, mal_family):
    # Create kav file to store kav scan result, and save kav file into nkrepo.
    f_kav = DIR_REPO + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256 + ".kav"
    #print(f_kav)
    if os.path.exists(f_kav):
        return 0
        #os.remove(f_kav)
    with open(f_kav, "w") as f:
        t = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        f.write("{}, {}, {}, {}, {}, {}".format(t, result, algorithm, mal_class, mal_platform, mal_family))
        return 1

def read_log():
    ''' Read Kaspersky raw scan log file and extract samples detection information.
The extracted information is stored in kav_results.txt already.'''

    list_scan_result = [] 
    pattern_sha256 = r'[a-f0-9]{64}'
    # Algorithm:Class.Platform.Famliy.Variant
    pattern_result = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)(\.[a-zA-Z0-9-]*)*'
    scan_list = []
    _n = 0
    with open(KAV_LOG, mode = "r", encoding = "utf-8") as f:
        for _l in f.readlines():
            #print(_l)
            # Searching sha256 string
            sha256 = re.search(pattern_sha256, _l)
            if not sha256:
                # If no sha256 string found, continue to next kav log
                continue
            sha256 =  sha256.group()
            #print(sha256)
            # Searching Kaspersky scan result
            result = re.search(pattern_result, _l)
            if not result:
                # If no Kaspersky result is found, continue to next kav log
                continue
            # 1. detection method, such as HEUR...
            if result.group(1): 
                algorithm = result.group(1)
            else:
                algorithm = ""
            # 2. malware class, such as Trojan, Worm,...
            mal_class = result.group(2)
            # 3. malware running platform, such as Win32, Script, ...
            mal_platform = result.group(3)
            # 4. malware family, such as  
            mal_family = result.group(4)
            # 5. malware variant information
            #mal_variant = result.group(5)
            # 6. detection result: Algorithm:Class.Platform.Family.Variant
            result = result.group()

            scan_result = "{}, {}, {}, {}, {}".format(sha256, mal_class, mal_platform, mal_family, result)
            print(scan_result)

            list_scan_result.append(scan_result)


            _n = _n + 1
            print("{}: {} {}".format(_n, sha256, result))

    list_scan_result = set(list_scan_result) # remove duplicated results
    save_csv(list_scan_result)
    return
 
def main():
    read_log() # Read Kaspersky scan result and extract class and family information
    
if __name__ == "__main__":
    main()
