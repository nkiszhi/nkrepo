#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil
from multiprocessing import Pool

#############################################
# Copy unlabeled samples into a temp folder.
#############################################

DIR_TEMP = "/nkrepo/temp/"
FILE_SHA256 = "sha256.txt"


def main():
    f = open(FILE_SHA256, "r",encoding='gb18030')
    _n = 0
    while True:
        l = f.readline().strip()
        if l:
            _f = DIR_TEMP + l
            if os.path.exists(_f):
                _n = _n + 1
                print("{}: {}".format(_n, _f))
                os.remove(_f)
        else:
            break

if __name__=="__main__":
    main()
