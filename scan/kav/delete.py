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

'''Delete scaned samples listed in the kaspersky log file.'''

LOG_KAV = "kav.log"

def delete_sample(line):
    # extract sample path 
    pattern_vs = r'([a-fA-F0-9\\:]*\\VirusShare_[0-9]{5}\\VirusShare_[a-f0-9]{32})'
    file_vs = re.search(pattern_vs, line)
    if not file_vs:
        return
    file_vs = file_vs.group(1)
    if os.path.exists(file_vs):  
        os.remove(file_vs)
        print(file_vs)

def read_log(file_log):
    ''' Read Kaspersky log file and extract scan result.'''
    list_result = []
    _n = 0
    # Read kav log
    print(file_log)
    with open(file_log, mode="r", encoding="utf-8") as f:
        list_result = f.readlines()
    list_result = [x.strip() for x in list_result]

    # File name is VirusShare_MD5 which is more than 43 characters.
    list_result = list(filter(lambda x: len(x) > 100, list_result))

    # delete scaned samples
    list_result = [delete_sample(x) for x in list_result]

def main():
    read_log(LOG_KAV)

if __name__ == "__main__":
    main()
