#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib, os
from multiprocessing import Pool

HEXSTRING = "0123456789abcdef"
MD5_DIR = "/nkrepo/DATA/md5/"
SHA256_DIR = "/nkrepo/DATA/sha256/"

def get_hash(f_path:str, hash_method) ->str :
    if not os.path.exists(f_path):
        print("[!]: {} is not existed\n".format(f_path))
        return ''
    h = hash_method()
    with open(f_path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def get_md5(f_path: str) -> str:
    ''' Get file MD5 hash value '''
    return get_hash(f_path, hashlib.md5)


def get_sha256(f_path: str) -> str:
    ''' Get file SHA256 hash value '''
    return get_hash(f_path, hashlib.sha256)

def init_md5_repo():
    ''' Initialize md5 repo using 4-tier storage structure '''
    for i in HEXSTRING:
        for j in HEXSTRING:
            for k in HEXSTRING:
                for l in HEXSTRING:
                    # subfolders at the 4th level
                    d = MD5_DIR + "/" + i + "/" + j + "/" + k + "/" + l
                    if not os.path.exists(d):
                        os.makedirs(d)

def create_md5_files():
    # Create md5 files in md5 repo
    p = Pool(2)
    folder_list = []
    for i in HEXSTRING:
        for j in HEXSTRING:
            for k in HEXSTRING:
                for l in HEXSTRING:
                    # subfolders at 4th level
                    subfolder = SHA256_DIR + "/" + i + "/" + j + "/" + k + "/" + l
                    folder_list.append(subfolder)
    #for i in HEXSTRING:
    #    for j in HEXSTRING:
    #        for k in HEXSTRING:
    #            # subfolders at 4th level
    #            subfolder = SHA256_DIR + "/0/" + i + "/" + j + "/" + k
    #            folder_list.append(subfolder)
    p.map(worker, folder_list)
    

def worker(folder):
    list_all = os.listdir(folder)
    _n = 0
    for f in list_all:
        if len(f) != 64:
            continue
        _n = _n + 1
        sha256 = f
        f_sha256 = folder + "/" + f
        md5 = get_md5(f_sha256) 
        f_md5 = MD5_DIR + md5[0] + "/" + md5[1] + "/" + md5[2] + "/" + md5[3] + "/"  + md5
        if os.path.exists(f_md5):
            continue
        with open(f_md5, 'w') as f:
            f.write(sha256)
        #print("[o]: {}".format(f_md5))
    print("[o]: {} ## {}".format(folder, _n))

def main():
    #init_md5_repo() # Initialize MD5 4-tier storage structure
    create_md5_files() # Create md5 files according to sha256 files


if __name__=="__main__":
    main()
