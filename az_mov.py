#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Move samples from samples folder storing VirusShare.com samples
into the repo in four-level storing architecture.""" 

from __future__ import print_function
import os
import shutil
from time import sleep

AZ_RESULTS = "az_results"
REPO = "DATA"


def mov_az_results():
    n_del = 0 #The number of moved apk files
    n_mov = 0 #The number of deleted apk files

    files = os.listdir(AZ_RESULTS)
    if not files:
        return
    n_samples = len(files) 
    print("[o]: Moving {} samples in {}".format(n_samples, AZ_RESULTS))
    for f in files:
        if len(f) != 152:
            continue
        src_path = "{}/{}".format(AZ_RESULTS, f)
        #sha256 = os.popen("sha256sum {}".format(src_path)) 
        #sha256 = ((sha256.read()).split())[0]
        sha256 = f.split("=")[-1].lower()
        if len(sha256) != 64:
            continue
        i = sha256[0] # first level
        j = sha256[1] # second level
        k = sha256[2] # third level
        l = sha256[3] # fourth level
        dst_path = "{}/{}/{}/{}/{}/{}".format(REPO,i,j,k,l,sha256)
        #print(src_path)
        #print(dst_path)
        if os.path.exists(dst_path):
            # Delete duplicated samples
            os.remove(src_path)
            n_del = n_del + 1
            print("[i]: Deleted duplicated sample {}".format(f))
        else:
            shutil.move(src_path, dst_path)
            print("{}: {}".format(n_mov, f))
            n_mov = n_mov + 1
    
    print("[o] {} new samples are added into repo.".format(n_mov))
    print("[i] {} duplicated samples are deleted.".format(n_del))

def main():
    mov_az_results()

if __name__ == "__main__":
    main()
