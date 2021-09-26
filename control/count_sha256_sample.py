#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Count the number of all malware samples in the repo."""

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"


import os
from time import gmtime, strftime
from greet import greet
from multiprocessing import Pool

DIR_DATA = "../DATA/sha256/"

def worker(folder):
    _n = 0
    list_all = os.listdir(folder)
    for f in list_all:
        if len(f) == 64:
            _n += 1
    return _n

def main():
    greet()
    list_dir = []
    hex_string = "0123456789abcdef"
    p = Pool(200)
    print("\n启动200个进程正在统计样本数量\n")
    _count = []
    
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                for l in hex_string:
                    folder = DIR_DATA + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    folder = os.path.abspath(folder)
                    list_dir.append(folder)
    _count = p.map(worker, list_dir)
    print("计算机病毒样本库有样本 {} 个.".format(sum(_count)))
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print()


if __name__ == "__main__":
    main()
