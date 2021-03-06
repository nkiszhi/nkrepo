#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""

from __future__ import print_function
from multiprocessing import Pool
import os
import datetime

""" get file type, size, time information in a specific folder"""
def worker(folder):
    print("Start {}".format(folder))
    _n = 0
    list_all = os.listdir(folder)
    list_f_info = []
    for f in list_all:
        if len(f) == 64:
            sha256 = str(f)
            f = folder + f
            #print(f)
            str_cmd = "file {}".format(f)
            f_type = os.popen(str_cmd).read().strip().split("/")[-1]
            f_type = f_type.split(":")[-1]
            f_size = os.stat(f).st_size
            f_mtime = os.stat(f).st_mtime
            f_time = datetime.datetime.fromtimestamp(f_mtime)
            f_time = f_time.strftime("%Y-%m-%d %H:%M:%S")
            f_info = sha256 + ", " + str(f_size) + ", " + str(f_time) + ", " + str(f_type)
            #print(f_info)
            list_f_info.append(f_info)
            _n += 1
    f_csv = folder + "f_info.csv"
    with open(f_csv, "w") as f:
        for item in list_f_info:
            f.write("{}\n".format(item))
    print("Finished {}: {}".format(folder, len(list_f_info)))
    return _n

def main():
    list_dir = []
    hex_string = "0123456789abcdef"
    p = Pool(200)
    _count = []
    n = 0
    
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                for l in hex_string:
                    folder = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    f_csv = folder + "f_info.csv"
                    if os.path.isfile(f_csv):
                        continue
                    n = n + 1
                    print("{} : {}".format(n, folder))
                    list_dir.append(folder)
    _count = p.map(worker, list_dir)
    print(_count)

if __name__ == "__main__":
    #worker("./DATA/0/0/0/0/")
    main()
