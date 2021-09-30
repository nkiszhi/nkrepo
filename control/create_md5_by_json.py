#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib, os
import json
import shutil
from greet import greet
from multiprocessing import Pool

HEXSTRING = "0123456789abcdef"
MD5_DIR = "/nkrepo/DATA/md5/"
SHA256_DIR = "/nkrepo/DATA/sha256/"
DIR_SHA256 = "../DATA/sha256/"
DIR_MD5 = "../DATA/md5/"
DIR_TEMP = "./TEMP"

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

    
def create_md5_file_by_json():
    n_json = 0     # The number of json files 
    n_error = 0    # The number of json files whose response code is 0 
    n_created = 0  # The number of created md5 files
    for i in HEXSTRING:
        for j in HEXSTRING:
            for k in HEXSTRING:
                for l in HEXSTRING:
                    # subfolders at 4th level
                    folder = DIR_SHA256 + "/" + i + "/" + j + "/" + k + "/" + l + "/"
                    folder = os.path.abspath(folder) + "/"
                    #print(folder)
                    list_all = os.listdir(folder)
                    for f in list_all:
                        if f[-5:] != ".json":
                            continue
                        n_json = n_json + 1
                        #print(f)
                        f_json = folder + f 
                        f_json = os.path.abspath(f_json)
                        #print(f_json)
                        with open(f_json, "r") as f:
                            dict_json = json.load(f)

                        if len(dict_json.keys()) == 2:
                            response_code = dict_json["results"]["response_code"] 
                            if not response_code:
                                print("[!] Response Code is O: {}".format(f_json))
                                n_error = n_error + 1
                                dst_json = DIR_TEMP + "/" + f 
                                dst_jsonn = os.path.abspath(dst_json)
                                shutil.move(f_json, dst_json) 
                                continue
                        else:
                            response_code = dict_json["response_code"] 
                            if not response_code:
                                print("[!] Response Code is O: {}".format(f_json))
                                n_error = n_error + 1
                                dst_json = DIR_TEMP + "/" + f 
                                dst_jsonn = os.path.abspath(dst_json)
                                shutil.move(f_json, dst_json) 
                                continue
                         
                        if len(dict_json.keys()) == 2:
                            sha256 = dict_json["results"]["sha256"]
                            md5 = dict_json["results"]["md5"]
                        else:
                            sha256 = dict_json["sha256"]
                            md5 = dict_json["md5"]
                        f_md5 = DIR_MD5 + md5[0] + "/" + md5[1] + "/"  + md5[2] + "/" + md5[3] + "/" + md5 
                        if os.path.exists(f_md5):
                            continue
                        f_md5 = os.path.abspath(f_md5)
                        with open(f_md5, "w") as f:
                            f.write(sha256)
                        n_created = n_created + 1
                        print("{}:\n SHA256 {}\n MD5 {}".format(f_md5, sha256, md5)) 
                        print()
                        print("{} json file.\n{} json files with 0 as response code \n{} md5 files are created\n".format(n_json, n_error, n_created))

                        return



                        

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
    greet()
    create_md5_file_by_json()

if __name__=="__main__":
    main()
