#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""

import os
import datetime
from multiprocessing import Pool

def greet():
<<<<<<< HEAD:control/count_samples.py
    print("\t\t**********************************************************")
    print("\t\t**                                                      **")
    print("\t\t**           Cyber攻击代码样本库                        **")
    print("\t\t**                                                      **")
    print("\t\t**********************************************************")
=======
    print("\t******************************************")
    print("\t**                                      **")
    print("\t**           计算机病毒样本库           **")
    print("\t**                                      **")
    print("\t******************************************")
>>>>>>> fa646a3001f087b6718606fc6ad03747df0656ce:control/count_samples.py

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
                    folder = "/nkrepo/DATA/sha256/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                    list_dir.append(folder)
    _count = p.map(worker, list_dir)
<<<<<<< HEAD:control/count_samples.py
    print("Cyber 攻击代码样本库有样本 {} 个.".format(sum(_count)))
=======
    print("\t\t计算机病毒样本库有样本 {} 个.".format(sum(_count)))
    print(datetime.datetime.now())
>>>>>>> fa646a3001f087b6718606fc6ad03747df0656ce:control/count_samples.py
    print()


if __name__ == "__main__":
    main()
