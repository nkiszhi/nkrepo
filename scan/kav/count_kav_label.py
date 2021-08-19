#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""

from __future__ import print_function
from multiprocessing import Pool
import os

DIR_DATA = "/nkrepo/DATA/sha256/"
HEX_STRING = "0123456789abcdef"

def greet():
    print()
    print("\t******************************************")
    print("\t**                                      **")
    print("\t**     The Repo of Malware Samples      **")
    print("\t**              NKAMG                   **")
    print("\t**                                      **")
    print("\t******************************************")
    print()


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
    list_dir = []
    p = Pool(200)
    _count = []

    
    print("\tCounting kaspersky scan results with 200 processes...\n")
    
    for i in HEX_STRING:
        for j in HEX_STRING:
            for k in HEX_STRING:
                for l in HEX_STRING:
                    folder = DIR_DATA + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    list_dir.append(folder)

    _count = p.map(worker, list_dir)
    print("\tThere are {} Kaspersky scan results in the repo.".format(sum(_count)))
    print()


if __name__ == "__main__":
    main()
