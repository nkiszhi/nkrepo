#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
from multiprocessing import Pool
from requests.api import get
from virus_total_apis import PublicApi as VirusTotalPublicApi
from datetime import datetime

KEY_FILE = "/nkrepo/scan/nkvt/keys.txt"
JSON_DIR = "/nkrepo/scan/nkvt/json_files/"
SAMPLE_DIR = "/nkrepo/DATA/sha256/"
LOG_FILE = "/nkrepo/scan/nkvt/log/vt_scan.log"
HEXSTRING = "0123456789abcdef"

N = 25000

def get_vt_result(vt_key, sha256, dir_results, n_all):
    vt = VirusTotalPublicApi(vt_key)
    resp = vt.get_file_report(sha256)
    if 'response_code' in resp.keys():
        if resp['response_code'] == 200:
            ret_json = json.dumps(resp, sort_keys=False, indent=4)
            save = open(dir_results + sha256 + '.json', 'w')
            save.write(ret_json)
            save.close()
            print(str(n_all + 1) + "[o]: 200, {}".format(sha256))
            return resp['response_code']
        else: 
            print(n_all, " has return but != 200")
            return 300
    else:
        print(n_all, "no return")
        return 300

def download(vt_key): 
    todo_list = get_todo_list()
    print("Get {} samples to scan".format(len(todo_list)))

    p = Pool(processes=20)
    print("Start virustotal query:")

    n_all = 0
    while len(todo_list) > 0:
        n_all += 1
        p.apply_async(get_vt_result, args=(vt_key, todo_list.pop(0), JSON_DIR, n_all))
    p.close()
    p.join()
        
    print("download finish")

def mov_json():
    ''' Move json files into repo '''
    jsons = os.listdir(JSON_DIR)
    print("Start moving {} json files into repo.".format(len(jsons)))
    _n = 0

    for item in jsons:
        if item.split(".")[-1] != "json":
            continue
        if len(item.split(".")[0]) != 64:
            continue
        if item.split(".")[-1] == "json" and len(item.split(".")[0]) == 64:
            _n += 1
            src_f = JSON_DIR + item
            dst_f = SAMPLE_DIR + item[0] + "/" + item[1] + "/" + item[2] + "/" + item[3] + "/" + item
            print("{}: {} to {}".format(_n, src_f, dst_f))
            shutil.move(src_f, dst_f)
            #return

    print("Moved {} json files.".format(_n))

def write_log():
    count_json = len(os.listdir(JSON_DIR))
    with open(LOG_FILE, "a+") as f:
        f.write(str(datetime.now()) + " downloaded " + str(count_json) + " json files\n")

def get_keys() -> list:
    key_list = []
    
    with open(KEY_FILE, "r") as t:
        key_list = t.readlines()
        for i in range(len(key_list)):
            key_list[i] = key_list[i].replace("\n","")
    return key_list

def get_todo_list() -> list:
    # get todo_list and return for virustotal to scan
    todo_list = []
    folder_list = []

    for i in HEXSTRING:
        for j in HEXSTRING:
            for k in HEXSTRING:
                for l in HEXSTRING:
                    folder_list.append(SAMPLE_DIR + i + "/" + j + "/" + k + "/" + l + "/")

    for folder in folder_list:
        file_list = os.listdir(folder)
        for f in file_list:
            if len(f) != 64:
                continue
            f_json = folder + f + ".json" # Virustotal scan result
            if os.path.exists(f_json):
                continue
            f_kav = folder + f + ".kav" # Kaspersky scan result
            if os.path.exists(f_kav):
                continue
            todo_list.append(f)
            if len(todo_list) >= N: # Everyday only 40000 virustotal queries
                return todo_list
    return todo_list
    

def main():
    key_list = get_keys()
    for key in key_list:
        if len(key) == 64:
            print(key)
            download(key)
            write_log()
            mov_json()


if __name__ == "__main__":
    main()
