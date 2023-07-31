#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# count_sha256.py : Count sha256 samples in the repo.
# location: nkrepo/control/count.py

import os
import argparse
from greet import greet
from multiprocessing import Pool

DIR_SHA256 = os.path.abspath("../DATA/sha256/") # The repo storing json files
STRING_HEX = "0123456789abcdef"
N_WORKER = 10

def worker(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if len(f) == 64: # The length of SHA256 file name is 64 bytes
            #print(f)
            _n += 1
    print(folder + ":" + str(_n))
    return _n

def main():
    list_folder = [DIR_SHA256 + \
            "/" + i + "/" + j + "/" + k + "/" + l + "/" + m + "/" \
            for i in STRING_HEX \
            for j in STRING_HEX \
            for k in STRING_HEX \
            for l in STRING_HEX \
            for m in STRING_HEX]
    print(list_folder[0])
    print(list_folder[-1])
    print(len(list_folder))
    p = Pool(N_WORKER)
    _count = []
    _count = p.map(worker, list_folder)
    n = sum(_count)
    print(n)
    print("There are {} samples in the repo.".format(n))

if __name__ == "__main__":
    main()
