#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import multiprocessing
import subprocess
import time
import requests
import gzip
import csv
import math
import wget
import re
import json
import multiprocessing
import threading
from threading import Thread, main_thread
from multiprocessing import Pool

# 下载文件
def downloadSamples(sample_download_path, apikey, sample_data_path, download_list) -> list :
    for download_file in download_list :
        filename = download_file.lower()
        file_path = sample_data_path + "/{}/{}/{}/{}".format(filename[0], filename[1], filename[2], filename[3])
        # check the path
        if os.path.exists(file_path) == False : os.makedirs(file_path)
        file_path += "/{}".format(filename)
        # download
        wget.download(sample_download_path.format(apikey, download_file), out = file_path)


class Androzoo :
    def __init__(self):
        self.__apikey = r'0d8564dc5820037584f50737244c3ccd296834568a389402d9427278274c1622'
        self.__list_download_path = "https://androzoo.uni.lu/static/lists/latest.csv.gz"
        self.__sample_download_path = 'https://androzoo.uni.lu/api/download?apikey={}&sha256={}'
        self.__list_file_path = "./latest.csv"
        self.__list_zip_path = "./latest.csv.gz"
        self.__sample_data_path = "./DATA"
        self.__download_subprocess = 30
        self.__downloadpool = None
    
    def updateVirusDB(self) :
        if not os.path.exists(self.__list_file_path) or (time.time() - os.path.getatime(self.__list_file_path)) / 86400 > 5 :
            # download the new list
            print("downloading the csv..")
            wget.download(self.__list_download_path, out = self.__list_zip_path)

            # unzip the new list and remove the zip
            list_zip = gzip.GzipFile(self.__list_zip_path)
            with open(self.__list_file_path, "wb") as new_list :
                new_list.write(list_zip.read())

        sha256_list = []
        print("parsing the list")
        # parse the list and get the sha256
        with open(self.__list_file_path) as list_file :
            for index, row in enumerate(list_file.readlines()) :
                if index == 0 : continue
                row = row.split(",")[0]
                filename = row.lower()
                if os.path.exists(self.__sample_data_path + "/{}/{}/{}/{}/{}.az".format(filename[0], filename[1], filename[2], filename[3], filename)) == False : sha256_list.append(row)

        file_num = math.ceil(len(sha256_list) / self.__download_subprocess)
        print("download the samples...")
        # download
        self.__downloadpool = Pool(self.__download_subprocess)
        for process_num in range(self.__download_subprocess) :
            self.__downloadpool.apply_async(func = downloadSamples, args = (self.__sample_download_path, self.__apikey, self.__sample_data_path, sha256_list[file_num * process_num : file_num * (process_num + 1) if file_num * (process_num + 1) < len(sha256_list) else len(sha256_list)]))
        self.__downloadpool.close()
        self.__downloadpool.join()
        self.__downloadpool = None
        print("complete")


if __name__ == "__main__":
    androo = Androzoo()
    androo.updateVirusDB()
