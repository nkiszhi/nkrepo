#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"

"""Init DATA folder with 4 levels subfolders."""

import os

HEX_STRING = "0123456789abcdef"

list_folder = []



def main():
    n_folders = 0
    for i in HEX_STRING:
        for j in HEX_STRING:
            for k in HEX_STRING:
                for l in HEX_STRING:
                    
                    folder = i + "/"+ j + "/"+ k+ "/" + l + "/"
                    list_folder.append(folder)
    with open("list_subfolder.txt", "w") as f:
        for folder in list_folder:
            f.write(folder + "\n")

    print("[o]: Created {} folders.".format(n_folders))

if __name__ == "__main__":
    main()

