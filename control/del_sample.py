#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import hashlib

DIR_MD5 = "../DATA/md5/"
DIR_SHA256 = "../DATA/sha256/"

md5_folder = "./DATA/md5/"
sha256_folder = "./DATA/sha256/"
list_samples = []


def greet():
   print("\t**********************************************************")
   print("\t**                                                      **")
   print("\t**           The Repo of Malware Samples                **")
   print("\t**                    by NKAMG                          **")
   print("\t**                                                      **")
   print("\t**********************************************************")


def delete(filename):
    n_del = 0
    n_non = 0
    with open(filename,"r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    for line in lines:
        if len(line) != 64:
            continue
        sha256 = line
        f_sha256 = DIR_SHA256 + "{}/{}/{}/{}/{}".format(sha256[0],sha256[1],sha256[2],sha256[3],sha256)
        if os.path.isfile(f_sha256):
            os.remove(f_sha256)
            n_del = n_del + 1
            print("{} : 删除样本  {}".format(n_del,  f_sha256))
        else:
            n_non = n_non + 1
            print("样本不存在  {}".format(f_sha256))

    print()
    greet()
    print("一共删除 {} 恶意代码样本".format(n_del))
    print("{} 恶意代码样本不存在".format(n_non))
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
