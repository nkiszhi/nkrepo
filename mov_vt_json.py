#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" move virustotal scan results stored in results folder into repo.""" 

from __future__ import print_function
import os
import shutil

VT_RESULTS = "vt_results"
REPO = "DATA"


def mov_vt_results():
    json_files = os.listdir(VT_RESULTS)
    for f in json_files:
        print(f)
        #print(f[0])
        #print(f[1])
        #print(f[2])
        #print(f[3])
        src_path = "{}/{}".format(VT_RESULTS, f)
        dst_path = "REPO/{}/{}/{}/{}/{}".format(REPO, f[0], f[1], f[2], f[3], f)
        #print(src_path)
        #print(dst_path)
        shutil.move(src_path, dst_path)

def main():
    mov_vt_results()

if __name__ == "__main__":
    main()
