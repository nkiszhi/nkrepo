#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Count the number of all malware samples in the repo."""

from multiprocessing import Pool
import os


DATA_DIR = "../../DATA/sha256/"

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
                    folder = DATA_DIR + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    folder = os.path.abspath(folder)
                    list_dir.append(folder)
    _count = p.map(worker, list_dir)
    print("\tThere are {} VirusTotal scan results in the repo.".format(sum(_count)))
    print()



if __name__ == "__main__":
    main()
