#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" move samples into repo.""" 

from __future__ import print_function
import os
import shutil

hex_string = "0123456789abcdef"

for i in hex_string:
    for j in hex_string:
        for k in hex_string:
            folder = "./samples/{}/{}/{}/".format(i,j,k)
            if not os.path.exists(folder):
                continue
            files = os.listdir(folder)
            if not files:
                continue
            print(folder)
            for f in files:
                if len(f) != 64:
                    print(f)
                    continue
                l = f[3]
                src_path = folder+f
                dst_path = "./DATA/{}/{}/{}/{}/{}".format(i,j,k,l,f)
                #print(src_path)
                #print(dst_path)
                if os.path.exists(dst_path):
                    # Delete duplicated samples
                    os.remove(src_path)
                    print("Delete: {}".format(f))
                    continue
                print(f)
                shutil.move(src_path, dst_path)
                #exit()
