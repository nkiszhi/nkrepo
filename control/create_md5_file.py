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
DIR_TEMP = "./TEMP/"

    
def create_md5_file_by_json():
    n_json = 0     # The number of json files 
    n_error = 0    # The number of json files whose response code is 0 
    n_created = 0  # The number of created md5 files
    for i in HEXSTRING:
        for j in HEXSTRING:
            for k in HEXSTRING:
                for l in HEXSTRING:
                    #### 1. Get folder
                    folder = DIR_SHA256 + "/" + i + "/" + j + "/" + k + "/" + l + "/"
                    folder = os.path.abspath(folder) + "/"

                    #### 2. Iterate folder 
                    list_all = os.listdir(folder)
                    for f in list_all:
                        #### 3. Find json file
                        if f[-5:] != ".json":
                            continue
                        file_name = f
                        n_json = n_json + 1
                        #print(f)
                        f_json = folder + f 
                        f_json = os.path.abspath(f_json)
                        #print(f_json)
                        #### 4. Read json file
                        with open(f_json, "r") as f:
                            dict_json = json.load(f)

                        #### 5. Check response code
                        if len(dict_json.keys()) == 2:
                            response_code = dict_json["results"]["response_code"] 
                        else:
                            response_code = dict_json["response_code"] 

                        #### 6. Move error json files to temp folder 
                        if not response_code:
                            print("[!] Response Code is O: {}".format(f_json))
                            n_error = n_error + 1
                            dst_json =  DIR_TEMP + file_name 
                            dst_jsonn = os.path.abspath(dst_json)
                            shutil.move(f_json, dst_json) 
                            continue
                        
                        #### 7. Get SHA256 and MD5 value 
                        if len(dict_json.keys()) == 2:
                            sha256 = dict_json["results"]["sha256"]
                            md5 = dict_json["results"]["md5"]
                        else:
                            sha256 = dict_json["sha256"]
                            md5 = dict_json["md5"]

                        #### 8. Check md5 file existence 
                        f_md5 = DIR_MD5 + md5[0] + "/" + md5[1] + "/"  + md5[2] + "/" + md5[3] + "/" + md5 
                        if os.path.exists(f_md5):
                            continue

                        #### 9. Create md5 file
                        f_md5 = os.path.abspath(f_md5)
                        with open(f_md5, "w") as f:
                            f.write(sha256)

                        n_created = n_created + 1
                        print("{}:\n SHA256 {}\n MD5 {}".format(f_md5, sha256, md5)) 
                        print()
                        print("{} json file.\n{} json files with 0 as response code \n{} md5 files are created\n".format(n_json, n_error, n_created))


def main():
    greet()
    #init_md5_repo() # Initialize MD5 4-tier storage structure
    create_md5_file_by_json()
    #create_md5_files() # Create md5 files according to sha256 files


if __name__=="__main__":
    main()
