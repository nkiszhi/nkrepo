#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""

from __future__ import print_function
from multiprocessing import Pool
import os

def greet():
    print("\t\t**********************************************************")
    print("\t\t**                                                      **")
    print("\t\t**           Cyber攻击代码样本库                        **")
    print("\t\t**                                                      **")
    print("\t\t**********************************************************")

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
                    folder = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    list_dir.append(folder)
    _count = p.map(worker, list_dir)
    print("Cyber 攻击代码样本库有样本 {} 个.".format(sum(_count)))



if __name__ == "__main__":
    main()
