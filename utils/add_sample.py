#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# add_sample.py : add a new sample into repo
# location: nkrepo/control/add_sample.py

import os
import shutil
import argparse
import hashlib
from multiprocessing import Pool

DIR_SHA256 = os.path.abspath("../data/sample/")
N_WORKER = 10

def get_sha256(file_sample):
    return hashlib.sha256(open(file_sample, "rb").read()).hexdigest()

def worker(file_sample):
    file_src = file_sample
    sha256 = get_sha256(file_sample)
    file_dst = DIR_SHA256 + "/" + sha256[0] + "/" +  sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" +  sha256[4] + "/" +sha256 
    if os.path.exists(file_dst):
        print("[!] {} Already existed! ".format(file_dst))
        return 0
    shutil.copy(file_src, file_dst)
    print("[o] {} Added.".format(file_dst))
    return 1

def add_sample(input_folder):

    input_folder = os.path.abspath(input_folder)

    list_file = os.listdir(input_folder)
    if not len(list_file):
        print("[!] The folder {} is empty!".format(input_folder)) 
        return 0

    list_file = [input_folder + "/" + x for x in list_file]
    
    p = Pool(N_WORKER)
    n = []
    n = p.map(worker, list_file)
    return sum(n)

def parseargs():
    parser = argparse.ArgumentParser(description = "Add samples into NKAMG malware repo.")
    parser.add_argument("-d", "--dir", help="The folder containing samples.", type=str, default="./temp")
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    if not os.path.exists(args.dir):
        print("[!] Folder {} is not exist!".format(args.dir))
        return False
    n = add_sample(args.dir)
    print("[o] Added {} samples into NKAMG malware repo.".format(n))

if __name__ == '__main__':
    main()

