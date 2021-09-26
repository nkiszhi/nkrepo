#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""

from __future__ import print_function
from multiprocessing import Pool
import os

DIR_DATA = "../DATA/sha256/"

def greet():
    print("\t\t******************************************")
    print("\t\t**                                      **")
    print("\t\t**     The Repo of Malware Samples      **")
    print("\t\t**              NKAMG                   **")
    print("\t\t**                                      **")
    print("\t\t******************************************")

def worker(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if len(f) == 64:
            json_f = folder + f + ".json" # VirusTotal label file
            kav_f = folder + f + ".kav"   # Kaspersky label file
            if os.path.exists(json_f):
                continue 
            if os.path.exists(kav_f):
                continue 
            _n = _n + 1

    return _n

def main():
    greet()
    list_dir = []
    hex_string = "0123456789abcdef"
    p = Pool(200)
    print("\n\t\tCounting samples without scan result using 200 processes.\n")
    _count = []
    
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                for l in hex_string:
                    folder = DIR_DATA + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    folder = os.path.abspath(folder)
                    list_dir.append(folder)
    _count = p.map(worker, list_dir)
    print("\t\tThere are {} samples without scan results.".format(sum(_count)))
    print()
    print()



if __name__ == "__main__":
    main()
