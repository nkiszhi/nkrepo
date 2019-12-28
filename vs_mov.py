#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Move samples from samples folder storing VirusShare.com samples
into the repo in four-level storing architecture.""" 

from __future__ import print_function
import os
import shutil
from time import sleep


def mov_samples():
    hex_string = "0123456789abcdef"
    dir_samples = "./samples/"
    n_delete = 0
    n_mov = 0

    files = os.listdir(dir_samples)
    if not files:
        return
    n_samples = len(files) 
    print("[o]: Moving {} samples in {}".format(n_samples, dir_samples))
    for f in files:
        if len(f) !=43 :
            continue
        src_path = dir_samples+f
        sha256 = os.popen("sha256sum {}".format(src_path))
        sha256 = ((sha256.read()).split())[0]
        #print(src_path)
        #print(sha256)
        i = sha256[0] # first level
        j = sha256[1] # second level
        k = sha256[2] # third level
        l = sha256[3] # fourth level
        dst_path = "./DATA/{}/{}/{}/{}/{}".format(i,j,k,l,sha256)
        #print(src_path)
        #print(dst_path)
        if os.path.exists(dst_path):
            # Delete duplicated samples
            os.remove(src_path)
            n_delete = n_delete + 1
            print("[i]: Deleted duplicated sample {}".format(f))
        else:
            shutil.move(src_path, dst_path)
            n_mov = n_mov + 1
    
    print("[o] {} new samples are added into repo.".format(n_mov))
    print("[i] {} duplicated samples are deleted.".format(n_delete))

def main():
    mov_samples()

if __name__ == "__main__":
    main()
