#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import hashlib
repo_folder = "./DATA/"

def greet():
   print("\t\t**********************************************************")
   print("\t\t**                                                      **")
   print("\t\t**           Cyber攻击代码库                            **")
   print("\t\t**                                                      **")
   print("\t\t**********************************************************")

def add(input_folder):
    files = os.listdir(input_folder)
    n_add = 0
    n_exist = 0
    for sample in files:
        src_path = input_folder +"/"+ sample
        with open(src_path, "rb") as f:
            bytes = f.read() # read entire file as bytes
            sha256 = hashlib.sha256(bytes).hexdigest();
        dst_path = repo_folder + "{}/{}/{}/{}/{}".format(sha256[0],sha256[1],sha256[2],sha256[3],sha256)
        if not os.path.isfile(dst_path):
            shutil.copy(src_path,dst_path)
            n_add = n_add + 1
            print("{}: 添加 {} 到 {}".format(n_add, src_path, dst_path))
        else:
            n_exist = n_exist + 1
            print("重复样本 {}".format(src_path))
    print()
    greet()
    print("一共添加 {} Cyber攻击样本".format(n_add))
    print("{} Cyber攻击样本是重复的".format(n_exist))
    print()
    print()



def parseargs():
    parser = argparse.ArgumentParser(description = "to add samples in my DATA")
    parser.add_argument("-d", "--dir", help="input one dir", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseargs()
    add(args.dir)

