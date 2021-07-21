# 导入网络代理对象
from main.Server import Client
# 导入指令管理器
from main.InstManager import InstManager
# 导入模板
from main.Models import ScriptModel
from main.Models import getFileType
from main.Models import sendWebState
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
from threading import Thread
from multiprocessing import Pool
from main.StatisticsManager import *

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


class Androzoo(ScriptModel) :
    def __init__(self, instmanager, rootpath):
        self.__apikey = r'0d8564dc5820037584f50737244c3ccd296834568a389402d9427278274c1622'
        self.__list_download_path = "https://androzoo.uni.lu/static/lists/latest.csv.gz"
        self.__sample_download_path = 'https://androzoo.uni.lu/api/download?apikey={}&sha256={}'
        self.__list_file_path = "./nkaz/data/latest.csv"
        self.__list_zip_path = "./nkaz/data/latest.csv.gz"
        self.__sample_data_path = rootpath
        self.__download_subprocess = 30
        self.__instmanager = instmanager
        self.__state = multiprocessing.Value("i", 0)
        self.__weakday = multiprocessing.Value("i", 3)
        self.__killworker = multiprocessing.Value("i", 0)
        self.__downloadpool = None

        # 注册指令，此处为例子
        # inst表示指令内容，targetfunc表示指令绑定的函数，targetscript表示目标脚本，description表示指令描述，level为指令等级
        instmanager.addInstruction(inst = "count", targetfunc = self.count, targetscript = "nkaz", description = "nkaz count", level = 0)
        instmanager.addInstruction(inst = "state", targetfunc = self.getStateF, targetscript = "nkaz", description = "nkaz state", level = 0)
        instmanager.addInstruction(inst = "resettime", targetfunc = self.changeUpdateDay, targetscript = "nkaz", description = "nkaz change update day", level = 0)
        instmanager.addInstruction(inst = "stop", targetfunc = self.stopMainloop, targetscript = "nkaz", description = "nkaz stop", level = 1)
        instmanager.addInstruction(inst = "start", targetfunc = self.startMainloop, targetscript = "nkaz", description = "nkaz start", level = 0)
        instmanager.addInstruction(inst = "updatetime", targetfunc = self.showUpdateTime, targetscript = "nkaz", description = "show nkaz update time", level = 0)

        # 运行主进程
        self.__azproc = None
        self.start()

    def showUpdateTime(self, client : type(Client), args : tuple) :
        day = self.__weakday.value
        if day == 0 : day = "Monday"
        elif day == 1 : day = "Tuesday"
        elif day == 2 : day = "Wednesday"
        elif day == 3 : day = "Thursday"
        elif day == 4 : day = "Friday"
        elif day == 5 : day = "Saturday"
        elif day == 6 : day = "Sunday"
        else : day = "Unkown. There may have some errors."
        client.sendMessageShow(day)

    # 终止
    def stopMainloop(self, client : type(Client), args : tuple) :
        if client.instructionConfirm("WARNING : This instruction will stop nkaz script. Input Y/y to confirm. [Y/N]", "y", "Y") :
            client.sendMessageShow("Nkaz is shutting down...")
            self.stop()
            client.sendMessageShow("Complete!")

    # 启动
    def startMainloop(self, client : type(Client), args : tuple) :
        if self.isNormal() : client.sendMessageShow("The script is running.")
        else : 
            client.sendMessageShow("Starting nkaz.")
            self.start()
            client.sendMessageShow("Complete!")
        
    # 进行计数
    def count(self, client : type(Client), args : tuple) :
        if self.__state.value != 0 : client.sendMessageShow("The download process is running, so the number of the files may be incorrect.")
        if "TotalFileNumber" in self.getStatistics().keys() : client.sendMessageShow(str(self.getStatistics()["TotalFileNumber"]))
        else : client.sendMessageShow("0")
    
    # 更改唤醒时间
    def changeUpdateDay(self, client : type(Client), args : tuple) :
        newday = ""
        while len(newday) != 1 or newday not in "0123456" :
            newday = client.getInput("Input the new updatetime(day).")
        self.__weakday.value = int(newday)

    # 获得状态
    def getStateF(self, client : type(Client), args : tuple) :
        client.sendMessageShow(self.getState())

    # 启动主进程
    def start(self) :
        sendWebState("nkaz", 2)
        self.__state = multiprocessing.Value("i", 0)
        self.__weakday = multiprocessing.Value("i", 3)
        self.__killworker = multiprocessing.Value("i", 0)
        self.__azproc = multiprocessing.Process(target = self.mainloop)
        self.__azproc.start()

    def killworker(self) :
        while True :
            if self.__killworker.value == 1 :
                if self.__downloadpool != None : 
                    self.__downloadpool.terminate()
                self.__killworker.value = 0
                break
            time.sleep(1)

    # 主进程
    def mainloop(self) :
        p = Thread(target = self.killworker)
        p.start()
        hasUpdate = False
        while True :
            # 检查是否到更新时间
            if hasUpdate == True and time.localtime().tm_wday != self.__weakday.value : hasUpdate = False
            if hasUpdate != False or time.localtime().tm_wday != self.__weakday.value : 
                # 等待
                time.sleep(1)
                continue
            # 执行更新
            if self.__state.value == 0 : self.__updateVirusDB()
            hasUpdate = True

    # 终止主进程
    def stop(self) : 
        self.__killworker.value = 1
        while self.__killworker.value == 1 : pass
        self.__azproc.kill()
        self.__state.value = 0
        self.__killworker.value = 0
        sendWebState("nkaz", 0)
    

    def isNormal(self) : 
        return self.__azproc.is_alive()

    # 获得状态信息
    def getState(self) -> str : 
        if not self.__azproc.is_alive() : return "stop"
        elif self.__state.value == 0 : return "Sleeping"
        elif self.__state.value == 1 : return "Downloading the new list..."
        elif self.__state.value == 2 : return "Unzipping the new list..."
        elif self.__state.value == 3 : return "Parsing the list..."
        elif self.__state.value == 4 : return "Downloading the new samples..."
        return "maybe there has some errors"

    # 检查主进程是否正在运行
    def isRunning(self) -> bool : 
        if self.__state.value == 0 or not self.__azproc.is_alive(): return False
        else : return True

    # 获得样本的统计信息，以字典的方式返回所有file.json的内容汇总
    def getStatistics(self) -> dict : 
        res = {}
        if os.path.exists(self.__sample_data_path + "/file.json") :
            with open(self.__sample_data_path + "/file.json") as f :
                res = json.loads(f.read())
        return res

    # 获得样本的总体数量
    def getCount(self) -> int : 
        res = self.getStatistics()
        if "TotalFileNumber" not in res.keys() : return 0
        else : return res["TotalFileNumber"]

    
    def __updateVirusDB(self) :
        sendWebState("nkaz", 1)
        self.__state.value = 1
        if not os.path.exists(self.__list_file_path) or (time.time() - os.path.getatime(self.__list_file_path)) / 86400 > 5 :
            # download the new list
            wget.download(self.__list_download_path, out = self.__list_zip_path)

            self.__state.value = 2
            # unzip the new list and remove the zip
            list_zip = gzip.GzipFile(self.__list_zip_path)
            with open(self.__list_file_path, "wb") as new_list :
                new_list.write(list_zip.read())

        sha256_list = []

        self.__state.value = 3
        # parse the list and get the sha256
        with open(self.__list_file_path) as list_file :
            for index, row in enumerate(list_file.readlines()) :
                if index == 0 : continue
                row = row.split(",")[0]
                filename = row.lower()
                if os.path.exists(self.__sample_data_path + "/{}/{}/{}/{}/{}.az".format(filename[0], filename[1], filename[2], filename[3], filename)) == False : sha256_list.append(row)
        self.__state.value = 4

        file_num = math.ceil(len(sha256_list) / self.__download_subprocess)
        # write dirtyfile
        if not os.path.exists(self.__sample_data_path) : os.makedirs(self.__sample_data_path)
        with open(self.__sample_data_path + "/dirtyfile.az.temp", "w+") as f :
            for temp in sha256_list : f.write(str(temp) + "\n")
        if not os.path.exists(self.__sample_data_path + '/dirtyfile') : os.makedirs(self.__sample_data_path + '/dirtyfile')
        os.popen("mv " + self.__sample_data_path + "/dirtyfile.az.temp " + self.__sample_data_path + "/dirtyfile/" + str(int(time.time())) + ".az")

        # download
        self.__downloadpool = Pool(self.__download_subprocess)
        for process_num in range(self.__download_subprocess) :
            self.__downloadpool.apply_async(func = downloadSamples, args = (self.__sample_download_path, self.__apikey, self.__sample_data_path, sha256_list[file_num * process_num : file_num * (process_num + 1) if file_num * (process_num + 1) < len(sha256_list) else len(sha256_list)]))
        self.__downloadpool.close()
        self.__downloadpool.join()
        self.__downloadpool = None
        self.__state.value = 0
        sendWebState("nkaz", 2)