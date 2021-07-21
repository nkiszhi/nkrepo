from __future__ import print_function, with_statement
# 导入网络代理对象
from main.Server import Client
# 导入指令管理器
from main.InstManager import InstManager
# 导入模板
from main.Models import ScriptModel
from main.Models import sendWebState
from main.StatisticsManager import updateTree

import threading
import datetime
import os
from time import sleep
import json
import argparse
import multiprocessing
from multiprocessing import Process, Pool, Lock, Value
from virus_total_apis import PublicApi as VirusTotalPublicApi
from datetime import datetime
import shutil


def out_get_vt_result(vtkey, sha256, dir_results,n_all,labeledList):
    vt = VirusTotalPublicApi(vtkey)
    resp = vt.get_file_report(sha256)
    if resp['response_code'] == 200:
        ret_json = json.dumps(resp, sort_keys=False, indent=4)
        save = open(dir_results + '/' + sha256 + '.json', 'w')
        save.write(ret_json)
        save.close()
        labeledList.append(sha256)
        # print(str(n_all)+"[o]: 200, {}".format(sha256))

    sleep(1)
    return resp['response_code']

def devporc(vtkey,willTableList,dir_results,labeledList):
    n_all = 0
    p = Pool(processes=6)
    # print("start download tables")
    for sha256 in willTableList:
        n_all = n_all + 1
        p.apply_async(out_get_vt_result, args=(vtkey, sha256, dir_results,n_all,labeledList))
    p.close()
    p.join()




class VirusTotal(ScriptModel) :
    def __init__(self, instmanager, nkazpath, nkvspath):
        self.nameList = ["NKAZ", "NKVS"]
        self.running=0
        self.sleeping=Value("i", 0)
        self.hasBeenDownload=0
        self.__vtproc=None
        self.n_process = 6
        self.azneedTable =Value("i",1)
        self.vsneedTable =Value("i",1)
        self.repofolder = [nkazpath, nkvspath]
        self.__weekday=None
        # 看当前正在移动的是哪个目录：-1：没有进行移动操作；0：在移动NKAZ ；1：在移动VS
        self.__nowMovingFolder=-1
        self.nowDownloading=-1


        # 注册指令，此处为例子
        # inst表示指令内容，targetfunc表示指令绑定的函数，targetscript表示目标脚本，description表示指令描述，level为指令等级
        instmanager.addInstruction(inst="start", targetfunc = self.startMainloop, targetscript = "nkvt",
                                   description = "nkvt start: download tables and mov them to correct diretory.", level = 0)
        instmanager.addInstruction(inst="stop", targetfunc=self.stopMainloop, targetscript="nkvt",
                                   description="nkvt stop", level=1)
        # instmanager.addInstruction(inst="count", targetfunc=self.getCountF, targetscript="nkvt",
        #                            description="json files nkvt has download", level=0)
        instmanager.addInstruction(inst="state", targetfunc=self.getStateF, targetscript="nkvt",
                                   description="nkvt state", level=0)
        instmanager.addInstruction(inst="move", targetfunc=self.getMoveF, targetscript="nkvt",
                                   description="nkvt move state", level=0)
        instmanager.addInstruction(inst="download", targetfunc=self.getDownloadingF, targetscript="nkvt",
                                   description="nkvt download state", level=0)

        # 运行主进程
        self.start()


    # 与client的主控接口函数
    def startMainloop(self, client: type(Client), args: tuple):
        state=self.start()
        if state:
            client.sendMessageShow("Start Success")
        else:
            client.sendMessageShow("nkvt is running")

    def stopMainloop(self, client : type(Client), args : tuple):
        state=self.stop()
        if state:
            client.sendMessageShow("Stop Success")
        else:
            client.sendMessageShow("nkvt has been stopped")

    def getStateF(self, client : type(Client), args : tuple) :
        client.sendMessageShow(self.getState())

    def getCountF(self, client : type(Client), args : tuple) :
        client.sendMessageShow(str(self.getCount()))

    def getMoveF(self, client : type(Client), args : tuple) :
        client.sendMessageShow(self.getMovingState())

    def getDownloadingF(self, client : type(Client), args : tuple) :
        client.sendMessageShow(self.getDownloadingState())





    # 启动主进程
    def start(self):
        if self.running ==1:
            return 0
        self.running = 1
        sendWebState("nkvt",1)
        self.startloop()
        return 1


    # 终止主进程
    def stop(self) :
        if self.running ==0:
            return 0

        self.__vtproc.kill()

        self.sleeping.value = 0
        self.running = 0
        sendWebState("nkvt",0)
        self.__weekday=None
        return 1

    # 获得状态信息
    def getState(self) -> str:
        if self.running==1 and self.sleeping.value == 1:
            return "nkvt is sleeping"
        if self.running==1 and self.sleeping.value==0:
            return "nkvt is working"
        return "nkvt has been stopped"

    def getMovingState(self) -> str:
        if self.getState()=="nkvt is working":
            if self.__nowMovingFolder==-1:
                return "moving is finished"
            else:
                return "now is moving labels"
        elif self.getState()=="nkvt is sleeping":
            return "nkvt is sleeping"
        else: return "nkvt has been stopped"
    
    def getDownloadingState(self) -> str:
        if self.getState()=="nkvt is working":
            if self.nowDownloading==-1:
                return "not download"
            elif self.nowDownloading==0:
                return "downloading nkaz"
            else:
                return "downloading nkvs"
        elif self.getState()=="nkvt is sleeping":
            return "nkvt is sleeping"
        else: return "nkvt has been stopped"

    # 检查主进程是否正在运行
    def isRunning(self) -> bool :
        if self.running==1:
            return True
        else:
            return False

    # 检查主进程的运行是否正常
    def isNormal(self) -> bool :
        if self.running==1:
            return True
        else:
            return False

    # 获得样本的统计信息，以字典的方式返回所有file.json的内容汇总
    def getStatistics(self) -> dict :
        return {}

    # 获得已经打过标签的样本的总体数量
    def getCount(self) -> int :
        return self.hasBeenDownload


    def startloop(self):
        self.__vtproc = multiprocessing.Process(target=self.vtStart)
        self.__vtproc.start()








    # 以下是nkvt的主功能程序
    def vtStart(self):
        while True:
            if self.isTomorrow():
                # 唤醒并向网站发送唤醒状态
                self.sleeping.value = 0
                sendWebState("nkvt",1)

                
                # 更新移动状态
                self.__nowMovingFolder=0
                self.movFiles()
                self.__nowMovingFolder=-1


                # 调整当前下载状态的回显
                self.nowDownloading=1
                self.downloadTables(1)
                self.nowDownloading=-1
                self.nowDownloading=0
                self.downloadTables(0)
                self.nowDownloading=-1
                if self.azneedTable.value == 0 and self.vsneedTable.value == 0:
                    self.sleeping.value = 1
                    sendWebState("nkvt",2)
                    # print("nothing need to download, self.sleeping->1, sleep")
                    sleep(4 * 60 * 60)
                else:
                    # print("today's labels download finish. Good night.")
                    self.sleeping.value = 1
                    sendWebState("nkvt",2)
                    sleep(4 * 60 * 60)
            else:
                self.sleeping.value=0
                sendWebState("nkvt",1)
                self.movFiles()
                self.sleeping.value=1
                sendWebState("nkvt",2)
                sleep(4*60*60)

    def downloadTables(self,nameNumber):
        dirtyPath=self.repofolder[nameNumber]+"/dirtyfile"
        if not os.path.exists(dirtyPath) : os.makedirs(dirtyPath)
        dirtyList=os.listdir(dirtyPath)
        lastDirtyfile=dirtyPath + "/0."+self.nameList[nameNumber][-2:].lower()
        willTableList=[]
        currentReadlist=[]
        if len(dirtyList)!=0:
            # 设置当前需要移动的状态，是在移动哪个
            if nameNumber == 0: self.azneedTable.value = 1
            else: self.vsneedTable.value = 1

            labeledList = []

            if os.path.exists(lastDirtyfile):
                with open(lastDirtyfile,"r") as l:
                    lastTemp = l.readlines()
                for la in lastTemp:
                    # 判断这个sha256是否已经存在，不在就加入
                    la=la.replace("\n","").lower()
                    if not os.path.exists(self.repofolder[nameNumber]+"/"+ la[0]+"/"+ la[1]+"/"+ la[2]+"/"+ la[3]+ "/"+la+".json"):
                        willTableList.append(la)
                    else :
                        with open(self.repofolder[nameNumber]+"/"+ la[0]+"/"+ la[1]+"/"+ la[2]+"/"+ la[3]+ "/"+la+".json", "r") as f :
                            dat = json.loads(f.read())
                            if len(dat.keys()) <= 1 and ("Androzoo" in dat.keys() or "VirusShare" in dat.keys()) : willTableList.append(la.replace("\n", "").lower())
                            else : labeledList.append(la)
                # readlist.append(lastDirtyfile)

            while len(willTableList)<=22000:
                for i in dirtyList:
                    if i[-2:] == self.nameList[nameNumber][-2:].lower():
                        temp = None
                        with open(dirtyPath + "/" + i, "r") as f:
                            temp = f.readlines()
                        for sha in temp:
                            sha=sha.replace("\n","").lower()
                            if not os.path.exists(self.repofolder[nameNumber]+"/"+ sha[0]+"/"+ sha[1]+"/"+ sha[2]+"/"+ sha[3]+ "/"+sha+".json"):willTableList.append(sha.replace("\n", "").lower())
                            else :
                                with open(self.repofolder[nameNumber]+"/"+ sha[0]+"/"+ sha[1]+"/"+ sha[2]+"/"+ sha[3]+ "/"+sha+".json", "r") as f :
                                    dat = json.loads(f.read())
                                    if len(dat.keys()) <= 1 and ("Androzoo" in dat.keys() or "VirusShare" in dat.keys()) : willTableList.append(sha.replace("\n", "").lower())
                                    else : labeledList.append(sha)
                        currentReadlist.append(dirtyPath + "/" + i)
                        # os.remove(dirtyPath + "/" + i)
            
            dir_results = "./nkvt/data/" + self.nameList[nameNumber]
            if not os.path.exists(dir_results) : os.makedirs(dir_results)
            file_key = "./nkvt/key.txt"
            if not os.path.exists(file_key):
                os.makedirs(file_key)
                with open(file_key,"w") as k:
                    k.write("193a7f895cf3370a5dc512fde5acddccf57c51da1b401f7b86e23b34638b8822\n")
            parser = argparse.ArgumentParser(prog="nkvt", description='Get SHA256 Virus Total scan results.')
            parser.add_argument("-k", "--keys", default=file_key, help="a file containing Virus Total keys",
                                type=argparse.FileType('r'))
            parser.add_argument("-r", "--results", default=dir_results,
                                help="a folder containing Virus Total scan results (default: ./results")
            args = parser.parse_args()
            list_key = [line.rstrip('\n') for line in args.keys]
            dir_results = args.results

            if os.path.exists(dir_results):
                list_result = os.listdir(dir_results)
                if list_result:
                    list_result = [x[:64] for x in list_result]
                    willTableList = set(willTableList) - set(list_result)
            vtkey = list_key[0]
            # print(vtkey)
            
            devporc(vtkey, willTableList, dir_results, labeledList)

            restList=list(set(willTableList)-set(labeledList))
            # 获得本次打标签结束后，剩下的还没有打的列表，放回到dirtyfile里
            with open(lastDirtyfile,"w") as f:
                for i in restList:
                    f.write(str(i) + "\n")
            # 删除文件
            for t in currentReadlist:
                os.remove(t)

            log_folder="./nkvt/data/log"
            if not os.path.exists(log_folder):os.makedirs(log_folder)
            with open(log_folder+ "/" +self.nameList[nameNumber]+"Download.log","a+") as log:
                today=str(datetime.today())
                count=str(len(labeledList))
                log.write(today+"\tDownload counts="+count)
            
            # 恢复下载状态
            self.nowDownloading=-1

            # 给处理函数传递已经打过的东西
            updateTree(labeledList,self.repofolder[nameNumber])

        else:
            if nameNumber == 0:
                self.azneedTable.value = 0
                # print("self.azneedTable=0")
            else:
                self.vsneedTable.value = 0
                # print("self.vsneedTable=0")



    def movFiles(self):
        # 将各自临时下载目录下的文件移动到对应目录下
        for nameNumber in range(2):
            # 源文件夹
            input_folder = "./nkvt/data"
            # 目标文件夹
            repo_folder = self.repofolder[nameNumber]# repo_folder + self.nameList[nameNumber] + "DATA/"

            # 看临时下载文件的nkvt的data文件里是否有已下载文件
            input_folder = input_folder +"/" +self.nameList[nameNumber]
            if not os.path.exists(input_folder) : os.makedirs(input_folder)
            files = os.listdir(input_folder)
            # print("[o]: Moving vt scan results in {}".format(input_folder))

            count = 0
            # 确定最终的目标文件夹repo_folder
            

            # 移动已下载的文件
            for sample in files:
                sha256 = sample[:-5].lower()
                # source
                src_path = input_folder + "/" +  sample
                # destination
                dst_path = repo_folder + "/{}/{}/{}/{}/{}".format(sha256[0], sha256[1], sha256[2], sha256[3], sample)
                # print(src_path)
                # print(dst_path)
                # 如果目标文件夹下已经存在某个json文件，那么删除掉，再移动一次
                if not os.path.exists(repo_folder + "/{}/{}/{}/{}".format(sha256[0], sha256[1], sha256[2], sha256[3])):
                    os.makedirs(repo_folder + "/{}/{}/{}/{}".format(sha256[0], sha256[1], sha256[2], sha256[3]))
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                shutil.move(src_path, dst_path)
                count += 1
                # print("[i]: {}  {}".format(count, sample))
            
            # 记录移动日志
            log_folder="./nkvt/data/log"
            if not os.path.exists(log_folder) : os.makedirs(log_folder)
            if count!=0:
                today=datetime.today()
                writeStr=str(today)+"\tmoving count="+str(count)+"\n"
                with open(log_folder+ "/" + self.nameList[nameNumber]+"Moving.log","a+") as log:
                    log.write(writeStr)
                
            
            # 当前目录moving结束，恢复状态
            self.__nowMovingFolder=-1

        # print("move finish")

    def isTomorrow(self):
        # 判断是否到了第二天，作为唤醒程序的条件
        today=datetime.isoweekday(datetime.today())
        if today!=self.__weekday:
            self.__weekday=today
            return 1
        else:
            return 0


    # def check(self):
    #     def worker(folder):
    #         json_list = []
    #         list_all = os.listdir(folder)
    #         for f in list_all:
    #             if f[-5:] == ".json":
    #                 json_list.append(f[:-5].lower())
    #         return json_list
    #
    #     # 主功能：将还没有打标签的加入到一个list
    #     # nameNumber是要打标签的目录标号，0:NKAZ  1:NKVS
    #     # 并写一个日志文件，对执行过的东西的记录
    #     def getList(nameNumber):
    #         hex_string = "0123456789abcdef"
    #         folder_list = []
    #         _temp_json_list = []
    #         _count_json_list = []
    #         un_list = []
    #         scanned_list = []
    #
    #         # 获取所有的目录列表
    #         # print("\t开始name整理目录列表")
    #         for i in hex_string:
    #             for j in hex_string:
    #                 for k in hex_string:
    #                     for l in hex_string:
    #                         folder = self.repofolder[nameNumber] + "/" + i + "/" + j + "/" + k + "/" + l + "/" #"./"+ self.nameList[nameNumber].lower() + "/data/" + i + "/" + j + "/" + k + "/" + l + "/"
    #                         folder_list.append(folder)
    #         # print("\t目录列表整理完毕")
    #
    #         # print("\t开始统计json列表")
    #         # 获取已经打过标签的列表
    #         for fold in folder_list:
    #             _temp_json_list.append(worker(fold))
    #         for item in _temp_json_list:
    #             _count_json_list.extend(item)
    #         self.hasBeenDownload+=len(_count_json_list)
    #         # print("\tjson列表整理完毕，共得到%d已打好的标签" % len(_count_json_list))
    #         # print(_count_json_list[:3])
    #
    #         # 获取之前已经读取过的json列表，防止重复加入
    #         # 但是存在有在临时下载目录下的文件，先对这些重新进行统计
    #         # 要统计此目录下已经下好了但是还没有move的json文件
    #         dataFolder = "./nkvt/data/" + self.nameList[nameNumber] + "/"
    #         # 原来存放要打目录的标签
    #         jsonTxt = "./nkvt/data/" + self.nameList[nameNumber] + "UnJson.txt"
    #
    #         # 获取txt中的list
    #         with open(jsonTxt, 'r') as ed:
    #             jsonListInTxt = ed.readlines()
    #         for i in range(len(jsonListInTxt)):
    #             jsonListInTxt[i]=jsonListInTxt[i].replace("\n", "").lower()
    #
    #         # 获取统计目录下已经下载了json的样本list
    #         downloadedList = []
    #         list_all = os.listdir(dataFolder)
    #         for sample in list_all:
    #             if sample[-5:] == ".json":
    #                 downloadedList.append(sample[:-5].lower())
    #         self.hasBeenDownload+=len(downloadedList)
    #
    #         # 去除掉已经下载过json文件的样本
    #         finalWriteList = []
    #         for i in jsonListInTxt:
    #             if (i not in downloadedList) and (i not in _count_json_list):
    #                 finalWriteList.append(i)
    #
    #         with open(jsonTxt, "w") as f:
    #             for i in finalWriteList:
    #                 f.write(i + "\n")
    #
    #         with open("./nkvt/data/" + self.nameList[nameNumber] + "UnJson.txt", 'r') as ed:
    #             scanned_list = ed.readlines()
    #         for i in range(len(scanned_list)):
    #             scanned_list[i].replace("\n", "")
    #         # print("\t读取完毕")
    #
    #
    #         '''以上内容是对已有的Unscaned的txt的一个修正
    #             不涉及到增加新的需要打标签的样本
    #             以下内容为统计新的样本目录'''
    #
    #
    #
    #         # print("\t开始收集20w个未打json的sample列表")
    #         # 遍历之前获取的目录列表，开始遍历找到还没有打标签的列表
    #         # 一次获取20w个未打标签样本的sha256
    #         n = 0
    #         flag = False
    #         IsAllScanned = False
    #         for folder in folder_list:
    #             list_all = os.listdir(folder)
    #             for sample in list_all:
    #                 if (len(sample) == 67) and (sample[-3:]==("."+self.nameList[nameNumber][-2:].lower())) and (sample not in _count_json_list) and (sample not in scanned_list):
    #                     un_list.append(sample[:64])
    #                     n = n + 1
    #                     # print("%d  %s" % (n,sample))
    #                 if n >= 200000:
    #                     flag = True
    #                     break
    #             if flag:
    #                 break
    #         if n < 200000:
    #             # 如果在少于20w的时候，就已经退出循环了，那就表示已经获取了所有的没有打标签的列表，都读取完毕了
    #             IsAllScanned = True
    #         # print("\t未打标签列表已得到，长度为：%d" % (len(un_list)))
    #         # print()
    #         # print("\t开始将列表更新到nameUnJson.txt中")
    #         with open("./nkvt/data/" + self.nameList[nameNumber] + "UnJson.txt", 'a+') as f:
    #             for jsons in un_list:
    #                 f.write("%s\n" % jsons)
    #
    #         # 写入日志文件
    #         with open("./nkvt/data/log/" + self.nameList[nameNumber] + "Scan.log", 'a+') as f:
    #             f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\tGet " + str(n) + " unscanned samples\t" + "All had been scanned:  " + str(IsAllScanned)+"\n")
    #
    #         # 返回是否已经获取全部未打标签的样本列表
    #         return IsAllScanned
    #
    #     # AZ
    #     getList(0)
    #     # VS
    #     getList(1)
