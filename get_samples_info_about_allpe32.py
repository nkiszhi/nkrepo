#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""

from __future__ import print_function
from multiprocessing import Pool
import os
import datetime
import sys,pefile,re,peutils
import pandas as pd


""" get pack information of PE32 in a specific folder"""
def worker(folder):
    print("Start {}".format(folder))
    _n = 0
    _m = 0
    list_all = os.listdir(folder)#该文件夹下所有病毒文件
    list_f_info = []
    for f in list_all:
        if len(f) == 64:#如果是病毒文件
            sha256 = str(f)
            f = folder + f
            #print(f)
            str_cmd = "file {}".format(f)
            f_type = os.popen(str_cmd).read().strip().split("/")[-1]
            f_type = f_type.split(":")[-1]
            if f_type.find("PE32") != -1:#找到了
                _n +=1
                try:
                    pe = pefile.PE(f)
                except AttributeError as e:
                    print(e)
                    check = None
                except pefile.PEFormatError as e:
                    print(f)
                    print(e)
                    check = None
                else:
                    signature  = peutils.SignatureDatabase("userdb.txt")
                    check = signature.match_all(pe,ep_only = True)
                    if check:
                        _m +=1
                f_info = sha256 + "," + str(check)
                #print(f_info)
                list_f_info.append(f_info)
    f_csv = folder + "f_pack_info.csv"
    with open(f_csv, "w") as f:
        for item in list_f_info:
            f.write("{}\n".format(item))
    print("Finished {}: {}".format(folder, len(list_f_info)))
    return {_n:_m}



###########################################################
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-

items = []
def worker1(f):
    if os.path.isfile(f):
        with open(f,'r') as f:
            for line in f:
                line_list = line.strip('\n').split(",")
                print(line_list)
                items.append(line_list)
        with open("f_info_del.csv","a+") as f:
             for item in items:
                 f.write("{}\n".format(item))
             items.clear()
        return 1
    else:
        return 0

hex_string = "0123456789abcdef"
n = 0
list_f = []

def worker2():
    if os.path.exists("f_info_del.csv"):
        os.remove("f_info_del.csv")
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                for l in hex_string:
                    f = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/f_pack_info.csv"#构造csv文件名
                    list_f.append(f)

    p = Pool(100)#构造100个进程池
    n = p.map(worker1, list_f)
    print("Finished {} f_pack_info.csv".format(sum(n)))

##########################################################3



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
                    folder = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/"#构造文件名
                    f_csv = folder + "f_pack_info.csv"
                    if os.path.isfile(f_csv):
                        continue
                    n = n + 1#标记数据集最后一层文件夹中中不存在f_pack_info.csv文件的数量
                    print("{} : {}".format(n, folder))
                    list_dir.append(folder)
    _count = p.map(worker, list_dir)
    #print(_count)
    #print(_packcount)
    worker2()

if __name__ == "__main__":
    #worker("./DATA/0/0/0/0/")
    main()
