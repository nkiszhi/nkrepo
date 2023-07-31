#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# add_sample.py : add a new sample into repo
# location: nkrepo/control/add_sample.py

import os
import shutil
import argparse
import hashlib
from greet import greet
from multiprocessing import Pool

DIR_SHA256 = os.path.abspath("../DATA/sha256/")
N_WORKER = 10

def get_sha256(file_sample):
    return hashlib.sha256(open(file_sample, "rb").read()).hexdigest()

def worker(file_sample):
    file_src = file_sample
    sha256 = get_sha256(file_sample)
    file_dst = DIR_SHA256 + "/" + sha256[0] + "/" +  sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" +  sha256[4] + "/" +sha256 
    if os.path.exists(file_dst):
        print("[!] Already existed \"{}\".".format(file_dst))
        return 0
    shutil.copy(file_src, file_dst)
    print("[OK] Added sample: \"{}\".".format(file_dst))
    return 1

def add_sample(input_folder):

    input_folder = os.path.abspath(input_folder)
    if not os.path.exists(input_folder):
        print("[X]: The folder \"{}\" is not exist".format(input_folder)) 
        return

    list_file = os.listdir(input_folder)
    list_file = [input_folder + "/" + x for x in list_file]
    list_sha256 = [get_sha256(x) for x in list_file]
    
    p = Pool(N_WORKER)
    n = []
    n = p.map(worker, list_file)
    print(sum(n)) 
    return n

def parseargs():
    parser = argparse.ArgumentParser(description = "Add samples into NKAMG malware repo.")
    parser.add_argument("-d", "--dir", help="The folder containing samples to add", type=str, default="TEMP")
    args = parser.parse_args()
    return args

def main():
    greet()
    args = parseargs()
    n = add_sample(args.dir)

if __name__ == '__main__':
    main()

