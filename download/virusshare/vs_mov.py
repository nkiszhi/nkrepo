#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vs_mov.py: mov unzipped samples to nkrepo
# location: nkrepo/download/nkvs

import hashlib, os, shutil
from multiprocessing import Pool

# nkrepo/DATA/sha256/
REPO_DATA = os.path.abspath("../../DATA/sha256/")
# nkrepo/download/nkvs/DATA/
VS_DATA = os.path.abspath("./DATA")
ZIP_FILE = "zip.txt"

def get_hash(file_path: str, hash_method) -> str:
    if not os.path.exists(file_path):
        print("[X] File does not exist: " + file_path)
        return ''
    h = hash_method()

    with open(file_path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b: break
            h.update(b)
    return h.hexdigest()

def get_md5(file_path: str) -> str:
    return get_hash(file_path, hashlib.md5)

def get_sha256(file_path: str) -> str:
    return get_hash(file_path, hashlib.sha256)

def mov_sha256(folder):
    list_file = os.listdir(folder)
    n = len(list_file)
    if not n:
        print("[X] {} is empty.\n".format(folder))
        return 0

    for item in list_file:
        # file name(43 characters): VirusShare_ffffe93aa825a99da6a7ac80e45f0209
        if len(item) != 43:
            continue
        sha256 = get_sha256(folder + "/" + item)
        src_f = folder + "/" + item
        # five-level folder
        dst_f = REPO_DATA + "/" + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256[4] + "/" + sha256
        shutil.move(src_f, dst_f)
        print("\tMove {} to {}".format(src_f, dst_f))

def main():
    list_folder = []
    with open(ZIP_FILE, "r") as f:
        list_folder = f.readlines()
    # remove ".zip" and add VS_DATA into path
    list_folder = [VS_DATA + "/" + x[:-4] for x in list_folder]
    p = Pool(5)
    p.map(mov_sha256, list_folder)

if __name__ == "__main__":
    main()


    
