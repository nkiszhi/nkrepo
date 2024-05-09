#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# init_repo.py : initial repo with 5-level folder
# location: nkrepo/control/init_repo.py

import os
import argparse

STRING_HEX = "0123456789abcdef"
DIR_DATA = os.path.abspath("../data/samples")

def main():
    #1. 创建基于sha256哈希值的5层恶意代码样本存储结构
    n = 0
    dir_sample = os.path.abspath("../data/samples")
    list_sha256_folder = [ dir_sample + "/" + i + "/" + j + "/" + k + "/" + l + "/" + m + "/" \
            for i in STRING_HEX \
            for j in STRING_HEX \
            for k in STRING_HEX \
            for l in STRING_HEX \
            for m in STRING_HEX]
    for folder in list_sha256_folder:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("[o] Create {}".format(folder))
            n = n + 1

    print("[o] Created {} folders".format(n))

    #2. 创建存储恶意域名的文件夹
    dir_trail = os.path.abspath("../data/trails")
    if not os.path.exists(dir_trail):
        os.makedirs(dir__trail)


if __name__ == "__main__":
    main()
