from main.Server import Client
from main.InstManager import InstManager
from main.Models import *
import sys
import os
import traceback
import json
import time

class ExceptionManager :
    def __init__(self, instManager, path = "./main/data/exp.json", targetscript = ["nkaz", "nkvt", "nkvs", "main"]) :
        self.__instManager = instManager
        self.__path = path
        self.__targetscript = targetscript
        self.__lastcheck = 0
        self.__expnum = 0
        self.__targetexpcount = {}
        self.removeLog()
        self.__resetTargetExceptionCount()
        self.__instRegister()

    def __logException(self, targetscript : str, filepath : str, linenum : int, funcname : str, inst : str, description : str, content : str) :
        # 信息统计
        if targetscript not in self.__targetscript : return
        else : self.__targetexpcount[targetscript] += 1

        # 写入log
        exp = (self.__expnum, targetscript, time.time(), filepath, linenum, funcname, inst, description, content)
        self.__expnum += 1
        with open(self.__path, "a") as f :
            f.write(json.dumps(exp) + '\n')

    # 重置统计数据
    def __resetTargetExceptionCount(self) :
        for temp in self.__targetscript :
            self.__targetexpcount[temp] = 0
    
    # 已查看异常
    def __clearNew(self) :
        self.__lastcheck = self.__expnum

    # 指令注册
    def __instRegister(self) :
        self.__instManager.addInstruction("error", targetfunc = self.checkError, targetscript = "main", description = "show errors and exceptions", level = 0)

    def __expstrGenerator(self, exp : tuple) :
        return exp[1] + ":" +\
                "\n\tTime: " + time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime(exp[2])) + \
                "\n\tFile: " + str(exp[3]) + \
                "\n\tLine: " + str(exp[4]) + \
                "\n\tFunc: " + str(exp[5]) + \
                "\n\tInst: " + str(exp[6]) + \
                "\n\tDesc: " + str(exp[7]) + \
                "\n" + str(exp[8]) + "\n"

    # 输出所有错误和异常
    def showAll(self, client : type(Client)) :
        if not os.path.exists(self.__path) or self.__expnum == 0: 
            self.removeLog()
            self.__resetTargetExceptionCount()
            client.sendMessageShow("No error or exception.")
            return
        with open(self.__path) as f :
            for expjson in f.readlines() :
                client.sendMessageShow(self.__expstrGenerator(json.loads(expjson)))
        
        self.__resetTargetExceptionCount()
        self.__clearNew()

    # 输出新的错误和异常
    def showNew(self, client : type(Client)) :
        if not os.path.exists(self.__path) or self.getNewCount() == 0: 
            self.removeLog()
            self.__resetTargetExceptionCount()
            client.sendMessageShow("No error or exception.")
            return
        with open(self.__path) as f :
            for expjson in f.readlines() :
                exp = json.loads(expjson)
                if exp[0] >= self.__lastcheck : client.sendMessageShow(self.__expstrGenerator(exp))

        self.__resetTargetExceptionCount()
        self.__clearNew()

    def checkError(self, client : type(Client), args : tuple) :
        if len(args) == 0 : 
            self.showNew(client)
            return
        elif len(args) == 1 :
            if args[0] == 'rm' : 
                self.removeLog()
                return
            if args[0] == 'num' : 
                client.sendMessageShow(str(self.getNewCount()))
                return
            elif args[0] == 'all' : 
                self.showAll(client)
                return
            elif args[0] == 'size' : 
                client.sendMessageShow(str(self.getlogSize()) + "KB")
                return
            elif args[0] == 'help' : 
                client.sendMessageShow("args help:\n  num   show count.\n  rm    delete all logs.\n  all   show all logs.\n  size  show the log file's size.\n  help  show help.\n (none) check new errors.")
                return

        client.sendMessageShow("args wrong.\n input \'error help\' for help")


    # 获得统计数据
    def getCount(self) -> dict:
        return self.__targetexpcount

    def getNewCount(self) -> int :
        return self.__expnum - self.__lastcheck

    def getlogSize(self) -> int :
        if os.path.exists(self.__path) :
            return os.path.getsize(self.__path) / 1024
        return 0

    # 重置log
    def removeLog(self) :
        if os.path.exists(self.__path) : os.remove(self.__path)
        self.__lastcheck = 0
        self.__expnum = 0

    # 异常处理
    def handle(self) :
        _, expinfo, exptraceback = sys.exc_info()

        # 问题描述
        description = "Exception: "
        for temp in expinfo.args :
            description += str(temp) + ', '

        # 生成系统提示信息
        content = "Traceback: \n"
        for temp in traceback.format_tb(exptraceback) : content += temp

        # 解析然后记录
        flag = False
        for filepath, linenum, funcname, inst in traceback.extract_tb(exptraceback) :
            paths = filepath.split("/")
            for target in self.__targetscript : 
                if target in paths : 
                    if flag : 
                        self.__logException(target, filepath, linenum, funcname, inst, description, content)
                        return
                    flag = True