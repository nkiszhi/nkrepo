#!/usr/bin/env python3
# -*-coding: utf-8 -*-

import argparse  
import os  
import hashlib  
import requests  
import time  
from configparser import ConfigParser  

# VT API Key列表
list_key = []
# 待扫描样本的sha256列表
list_sha256 = []
# 存储待扫描样本的文件夹
dir_sample = ""
# 存储VT扫描结果的文件夹
dir_result = ""

# 读取配置文件， 返回VT API Key列表
def read_config():  
    global list_key
    cp = ConfigParser()  
    cp.read('config.ini')  
    list_key = cp.get('key', 'KEY').split(',')  

def get_sha256():
    global list_sha256
    list_sample = []
    for root, _, files in os.walk(dir_sample):
        for file in files:
            list_sample.append(os.path.join(root, file))
    
    if list_sample:
        list_sha256 = [hashlib.sha256(open(s, 'rb').read()).hexdigest() for s in list_sample]
        return True
    return False 

def save_json(sha256, result):
    file_json = os.path.join(dir_result, sha256+".json")
    with open(file_json, 'w')as f:
        f.write(str(result))
    print(sha256+".json")

def send_request(key, sha256):
    url = f'https://www.virustotal.com/api/v3/files/{sha256}'  
    headers = {'x-apikey': key} 
    try:  
        response = requests.get(url, headers=headers, timeout=40)  
        response.raise_for_status()  
        return response.json()  
    except requests.exceptions.RequestException:
        print("[!] Get vt result failed!")
        #有异常就更换KEY
        return None

def get_vt_result():
    n_sample = len(list_sha256)
    print("[o] {} samples for vt scanning！".format(n_sample))
    n_key = len(list_key)
    print("[o] {} vt keys for using！".format(n_key))

    n = int(n_sample/400)+1
    for i in range(n):
        key = list_key[i]
        list_scan = list_sha256[400*i:400*i+399]
        for sha256 in list_scan:
            save_json(sha256, send_request(key, sha256))
            time.sleep(20)  # 假设每次请求之间需要间隔20秒

def main():  
    parser = argparse.ArgumentParser(description='Get VirusTotal scan report for files in a specific directory.')  
    parser.add_argument('-d', '--dir', type=str, help='Path to a directory containing files to check.')  # -folder 文件夹路径
    parser.add_argument('-r', '--result', type=str, default="./results/", help='Path to save the scan result.')     # -p 保存结果路径
    args = parser.parse_args()  
  
    # 读取VT API Key列表，保存到全局变量list_key中
    read_config()
    # 读取指定的保存扫描样本的文件夹
    global dir_sample, dir_result



    dir_sample = args.dir
    print(dir_sample)

    # 读取指定的保存VT扫描结果的文件夹
    dir_result = args.result   #没有指定路径就保存为默认路径

    if not os.path.exists(dir_result):
        os.makedirs(dir_result)

    if not dir_sample:
        parser.print_help()
        return

    get_sha256()
    
    get_vt_result()

if __name__ == "__main__":  
    main()
