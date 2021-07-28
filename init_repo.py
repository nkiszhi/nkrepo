#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Init DATA folder with 4 levels subfolders."""

from __future__ import print_function
import os

hex_string = "0123456789abcdef"
n_folders = 0

for i in hex_string:
    for j in hex_string:
        for k in hex_string:
            for l in hex_string:
                sha256_folder = "./DATA/sha256/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                md5_folder = "./DATA/sha256/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                if not os.path.exists(sha256_folder):
                    os.makedirs(sha256_folder)
                    n_folders = n_folders + 1
                    print("[o]: Create {}".format(sha256_folder))
                if not os.path.exists(md5_folder):
                    os.makedirs(md5_folder)
                    n_folders = n_folders + 1
                    print("[o]: Create {}".format(md5_folder))
print("[o]: Created {} folders.".format(n_folders))


