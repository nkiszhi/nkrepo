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

#############################################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""

hex_string="0123456789abcdef"

def get_all():
    n=0
    df= pd.DataFrame(columns=['sha256','pack']) 
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                for l in hex_string:
                    for m in hex_string:
                        f = "../DATA/" + i + "/" + j + "/" + k + "/" + l + "/" + m + "/f_pack_info.csv"
                        if os.path.isfile(f):
                            df1 = pd.read_csv(f,header=None,names=['sha256','pack'])
                            df = pd.concat([df,df1], ignore_index=True)
                            n +=1
                            print(n)
                        print(df)
    df.to_csv("pack_info.csv")

#############################################

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""




hex_string="0123456789abcdef"

def get_all_userdb():
    f = "pack_info.csv"#有数据集中所有csv中文件的信息
    df = pd.read_csv(f,index_col=0)
    df = df[df.pack !='None']
    df.reset_index(drop=True,inplace=True)
    print(df)
    df.to_csv("pack.csv")

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
                    for m in hex_string:
                        folder = "../DATA/" + i + "/" + j + "/" + k + "/" + l + "/" + m + "/"#构造文件名
                        f_csv = folder + "f_pack_info.csv"
                        if os.path.isfile(f_csv):
                            continue
                        n = n + 1#标记数据集最后一层文件夹中中不存在f_pack_info.csv文件的数量
                        print("{} : {}".format(n, folder))
                        list_dir.append(folder)
    _count = p.map(worker, list_dir)
    print(_count)
    #print(_packcount)
    get_all()
    get_all_userdb()

if __name__ == "__main__":
    #worker("./DATA/0/0/0/0/")
    main()

