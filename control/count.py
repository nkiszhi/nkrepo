#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"

"""Count the number of VirusTotal json files in the repo."""

import os
import argparse
from greet import greet
from multiprocessing import Pool

DIR_DATA = "../DATA/sha256/" # The repo storing json files
LIST_SUBFOLDER = "list_subfolder.txt"

def worker(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if f[64:] == ".kav":
            #print(f)
            _n += 1
    return _n

def main():
    greet()
    p = Pool(200)
    list_folder = []
    _count = []

    with open(LIST_SUBFOLDER, "r") as f:
        list_folder = f.readlines()
    list_folder = [os.path.abspath(DIR_DATA + x.strip()) for x in list_folder]

    _count = p.map(worker, list_folder)
    print("There are {} samples with Kaspersky scan result in the repo.".format(sum(_count)))

if __name__ == "__main__":
    main()
