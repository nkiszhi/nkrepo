#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Move samples from samples folder in  the three-level storing architecture
into the repo in four-level storing architecture.""" 

from __future__ import print_function
import os
import shutil
from time import sleep


def mov_samples():
    hex_string = "0123456789abcdef"
    n_delete = 0
    n_mov = 0
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                folder = "./samples/{}/{}/{}/".format(i,j,k)
                if not os.path.exists(folder):
                    continue
                files = os.listdir(folder)
                if not files:
                    continue
                print("[o]: Moving samples in {}".format(folder))
                for f in files:
                    if len(f) != 64:
                        #print(f)
                        continue
                    l = f[3]
                    src_path = folder+f
                    dst_path = "./DATA/{}/{}/{}/{}/{}".format(i,j,k,l,f)
                    #print(src_path)
                    #print(dst_path)
                    if os.path.exists(dst_path):
                        # Delete duplicated samples
                        os.remove(src_path)
                        n_delete = n_delete + 1
                        print("[i]: Deleted duplicated sample {}".format(f))
                        continue
                    shutil.move(src_path, dst_path)
                    n_mov = n_mov + 1
    
    print("[o] {} new samples are added into repo.".format(n_mov))
    print("[i] {} duplicated samples are deleted.".format(n_delete))

def main():
    while 1:
        mov_samples()
        sleep(120)

if __name__ == "__main__":
    main()
