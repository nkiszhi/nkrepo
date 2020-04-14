#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Unzip tar.gz files
The tar.gz files are stored at temp folder.
The extracted samples are stored at samples folder.
"""

from __future__ import print_function
import os
import shutil
from multiprocessing import Pool
from time import sleep


def job(f_zip, f_unzip):
    os.system("tar xvzf {} -C samples".format(f_zip)) 
    shutil.move(f_zip, f_unzip)
    
    return f

def main():
    files = os.listdir("az_zip")
    p = Pool(50)
    for f in files:
        if f[-7:] != ".tar.gz":
            continue
        f_zip = "az_zip/"+f
        f_unzip = "az_unzip/"+f
        p.apply_async(job, (f_zip, f_unzip, ))
    p.close()
    p.join()

if __name__ == "__main__":
    main()
