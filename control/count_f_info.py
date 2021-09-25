#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from multiprocessing import Pool
import os

def worker(f):
    if os.path.isfile(f):
        return 1
    else:
        return 0

hex_string = "0123456789abcdef"
n = 0
list_f = []

for i in hex_string:
    for j in hex_string:
        for k in hex_string:
            for l in hex_string:
<<<<<<< HEAD:control/count_f_info.py
                f = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/f_info.csv"
=======
                f = "./DATA/sha256/" + i + "/"+ j + "/"+ k+ "/" + l + "/f_pack_info.csv"
>>>>>>> fa646a3001f087b6718606fc6ad03747df0656ce:control/count_f_info.py
                list_f.append(f)

p = Pool(100)
n = p.map(worker, list_f)
print("Finished {} f_info.csv".format(sum(n)))


