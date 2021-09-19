#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
"""Init DATA folder with 4 levels subfolders."""

import os

DIR_SHA256 = "/nkrepo/DATA/sha256"
DIR_MD5 = "/nkrepo/DATA/md5"
HEX_STRING = "0123456789abcdef"


def main():
    n_folders = 0
    for i in HEX_STRING:
        for j in HEX_STRING:
            for k in HEX_STRING:
                for l in HEX_STRING:
                    sha256_folder = DIR_SHA256 + "/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    md5_folder = DIR_MD5 + "/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    if not os.path.exists(sha256_folder):
                        os.makedirs(sha256_folder)
                        n_folders = n_folders + 1
                        print("[o]: Create {}".format(sha256_folder))
                    if not os.path.exists(md5_folder):
                        os.makedirs(md5_folder)
                        n_folders = n_folders + 1
                        print("[o]: Create {}".format(md5_folder))
    print("[o]: Created {} folders.".format(n_folders))

if __name__ == "__main__":
    main()
