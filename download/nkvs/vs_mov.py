#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib, os, shutil
from multiprocessing import Pool

# file name(43 characters): VirusShare_ffffe93aa825a99da6a7ac80e45f0209

  
ROOT_PATH = "/nkrepo/download/nkvs/DATA/tmp/"
REPO_PATH = "/nkrepo/DATA/sha256/"

def get_hash(file_path:str,hash_method) ->str :
    # 参数说明：filepath是给定的要计算的文件的路径，包括自己的名称，如：/folder/file.txt
    if not os.path.exists(file_path):
        print("Not existed: " + file_path)
        return ''
    h = hash_method()
    with open(file_path,"rb") as f:
        while True:
            b=f.read(8192)
            if not b:break
            h.update(b)
    return h.hexdigest()


def get_md5(file_path: str) -> str:
    return get_hash(file_path, hashlib.md5)


def get_sha256(file_path: str) -> str:
    return get_hash(file_path, hashlib.sha256)


def mov_sha256(folder):
    #print("Moving {} \n".format(folder))
    
    files = os.listdir(folder)
    if len(files):
        print("{}: {} lefted.\n".format(folder, len(files)))
    #return len(files)

    for item in files:
        if len(item) != 43:
            continue 
        sha256 = get_sha256(folder + item)
        src_f = folder + item
        dst_f = REPO_PATH + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256
        shutil.move(src_f, dst_f)
        print("\tMove {} to {}".format(src_f, dst_f))

def main():
    dir_list = os.listdir(ROOT_PATH)
    mov_list = []
    _n = 0
    
    for item in dir_list:
        mov_list.append(ROOT_PATH + item + "/")
    
    p = Pool(100)
    _n = p.map(mov_sha256, mov_list)
    print("{} files lefted\n".format(sum(_n)))


if __name__=="__main__":
    main()
    #path_test = "/nkrepo/nkvs/DATA/tmp/VirusShare_00328/"
    #rename_sha256(path_test) 
