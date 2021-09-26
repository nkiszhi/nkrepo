#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"

"""Init DATA folder with 4 levels subfolders."""

import os

DIR_SHA256 = "../DATA/sha256"
DIR_MD5 = "../DATA/md5"
HEX_STRING = "0123456789abcdef"

def main():
    n_folders = 0
    for i in HEX_STRING:
        for j in HEX_STRING:
            for k in HEX_STRING:
                for l in HEX_STRING:
                    dir_sha256 = DIR_SHA256 + "/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    dir_sha256 = os.path.abspath(dir_sha256)
                    dir_md5 = DIR_MD5 + "/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    dir_md5 = os.path.abspath(dir_md5)
                    if not os.path.exists(dir_sha256):
                        os.makedirs(dir_sha256, exist_ok=True)
                        n_folders = n_folders + 1
                        print("[o]: Create {}".format(dir_sha256))
                    if not os.path.exists(dir_md5):
                        os.makedirs(dir_md5, exist_ok=True)
                        n_folders = n_folders + 1
                        print("[o]: Create {}".format(dir_md5))
    print("[o]: Created {} folders.".format(n_folders))

if __name__ == "__main__":
    main()
