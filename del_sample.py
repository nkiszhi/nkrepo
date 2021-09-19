#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import hashlib

md5_folder = "./DATA/md5/"
sha256_folder = "./DATA/sha256/"
list_samples = []


def greet():
   print("\t**********************************************************")
   print("\t**                                                      **")
   print("\t**           Cyber攻击代码库样本检索工具                **")
   print("\t**                                                      **")
   print("\t**********************************************************")


def delete(filename):
    n_del = 0
    n_non = 0
    with open(filename,"r") as f:
        for line in f:
            if len(line.strip('\n')) == 64:
                sha256 = line.strip('\n')
                dst_path = md5_folder + "{}/{}/{}/{}/{}".format(sha256[0],sha256[1],sha256[2],sha256[3],sha256)
                if os.path.isfile(dst_path):
                    os.remove(dst_path)
                    n_del = n_del + 1
                    print("{} : 删除样本  {}".format(n_del,  dst_path))
                else:
                    n_non = n_non + 1
                    print("样本不存在  {}".format( dst_path))

                dst_path = sha256_folder + "{}/{}/{}/{}/{}".format(sha256[0],sha256[1],sha256[2],sha256[3],sha256)
                if os.path.isfile(dst_path):
                    os.remove(dst_path)
                    n_del = n_del + 1
                    print("{} : 删除样本  {}".format(n_del,  dst_path))
                else:
                    n_non = n_non + 1
                    print("样本不存在  {}".format( dst_path))

    print()
    greet()
    print("一共删除 {} Cyber攻击样本".format(n_del))
    print("{} Cyber攻击样本不存在".format(n_non))
    print()
    print()



def parseargs():
    parser = argparse.ArgumentParser(description = "to delete samples in my DATA")
    parser.add_argument("-s", "--sha256", help="input a sha256.txt", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    delete(args.sha256)

if __name__ == '__main__':
    main()
