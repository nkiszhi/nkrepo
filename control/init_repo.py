#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# init_repo.py : initial repo with 5-level folder
# location: nkrepo/control/init_repo.py

import os
import argparse

STRING_HEX = "0123456789abcdef"
DIR_DATA = os.path.abspath("../DATA/")

def main():
    n = 0
    list_sha256_folder = [ DIR_DATA + "/sha256/" + i + "/" + j + "/" + k + "/" + l + "/" + m + "/" \
            for i in STRING_HEX \
            for j in STRING_HEX \
            for k in STRING_HEX \
            for l in STRING_HEX \
            for m in STRING_HEX]
    for folder in list_sha256_folder:
        print(folder)
        if os.path.exists(folder):
            continue
        os.makedirs(folder)
        n = n + 1
    print("Created {} sha256 subfolders".format(n))

if __name__ == "__main__":
    main()
