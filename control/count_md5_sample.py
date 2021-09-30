#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""

import os
from time import gmtime, strftime
from multiprocessing import Pool
from greet import greet

DIR_DATA = "../DATA/md5/"
LIST_SUBFOLDER = "list_subfolder.txt"

def worker(folder):
    _n = 0
    list_all = os.listdir(folder)
    #print(len(list_all))
    for f in list_all:
        if len(f) == 32:
            _n += 1
    return _n

def main():
    greet()
    list_folder = []
    p = Pool(200)

    print("\n启动200个进程正在统计样本数量\n")
    _count = []

    with open(LIST_SUBFOLDER, "r") as f:
        list_folder = f.readlines()
    list_folder = [os.path.abspath(DIR_DATA + x.strip()) for x in list_folder]
    
    _count = p.map(worker, list_folder)
    print("\t\t恶意代码样本库有样本 {} 个.".format(sum(_count)))
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print()


if __name__ == "__main__":
    main()
