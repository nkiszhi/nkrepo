#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool

count=0

def worker(route):
    _n = 0
    fileList=os.listdir(route)
    for item in fileList:
        if item.split(".")[-1]=="json":
            _n += 1
    return _n

folderList=[]
hexstring="0123456789abcdef"
for i in hexstring:
    for j in hexstring:
        for k in hexstring:
            for l in hexstring:
                folder="./DATA/"+i+"/"+j+"/"+k+"/"+l
                folderList.append(folder)

# folderList=["../t","../temp","../能独立完整运行的vs等"]
# print(folderList)
# worker(folderList[0])
# print(count)

print("正在开启200个进程进行统计json数目")
p=Pool(200)
_count = []
_count=p.map(worker,folderList)
print("There are {} json files in nkamg.".format(sum(_count)))