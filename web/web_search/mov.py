#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import argparse
import tarfile
import os
import shutil

DIR_SRC = "/home/RaidDisk/nkrepo/web/search/samples/"        # folder containing all samples
DIR_DST = "/home/RaidDisk/nkrepo/DATA/sha256/"        # folder containing all samples


def main():
    for i in os.listdir('samples'):
        file_src = DIR_SRC + i
        if not os.path.exists(file_src):
            continue
        file_dst = DIR_DST + i[0] + '/' +  i[1] + '/' + i[2] + '/' + i[3] + '/' + i 
        if os.path.exists(file_dst):
            continue
        shutil.copyfile(file_src, file_dst) # copy samples
        n = n + 1
        print("{}: {}".format(n, i))
    

if __name__ == "__main__":
    main()
