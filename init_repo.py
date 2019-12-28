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
                folder = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    n_folders = n_folders + 1
                    print("[o]: Create {}".format(folder))
print("[o]: Created {} folders for initing repo.".format(n_folders))

