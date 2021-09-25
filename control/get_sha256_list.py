#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"

""" Get a list of all VirusTotal.com scan results.
"""

import os

DIR_REPO = "/home/RaidDisk/nkrepo/DATA/sha256/0/"
FILE_SHA256 = "list_sha256.txt"

def main():
    hex_string = "0123456789abcdef"
    sample_count = 0
    list_sha256 = []
    
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                #for l in hex_string:
                folder = DIR_REPO + i + "/"+ j + "/"+ k + "/"
                print(folder)
                list_all = os.listdir(folder)
                for f in list_all:
                    if len(f) == 64:
                        list_sha256.append(f)
                        #print(f)
    
    with open(FILE_SHA256, 'w') as f:
        for i in list_sha256:
            f.write("{}\n".format(i))

if __name__ == "__main__":
    main()
