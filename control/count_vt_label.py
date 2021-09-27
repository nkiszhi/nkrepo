#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"

"""Count the number of VirusTotal json files in the repo."""

import os
from greet import greet
from time import gmtime, strftime
from multiprocessing import Pool

DIR_REPO = "../DATA/sha256/" # The repo storing json files

def worker(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if f[64:] == ".json":
            #print(f)
            _n += 1
    return _n

def main():
    greet()
    list_dir = []
    hex_string = "0123456789abcdef"
    p = Pool(200)
    _count = []
    
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                for l in hex_string:
                    folder = DIR_REPO + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    folder = os.path.abspath(folder)
                    list_dir.append(folder)
    _count = p.map(worker, list_dir)
    print("There are {} samples with VirusTotal scan result in the repo.".format(sum(_count)))
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))



if __name__ == "__main__":
    main()