#!/usr/bin/python
#-- coding:utf8 --
from abc import ABC, abstractmethod
import os
import json
import pefile
from main.data.hashdata import virusPossibleAPI, fileTypeList
import math
import struct

def getNKAZPath() :
    return '../NKVSDATA'

def getNKVSPath() :
    return '../NKVSDATA'

# 模板，用于限制接口
class ScriptModel(ABC) :
    # 启动主进程
    @abstractmethod
    def start(self) : pass

    # 终止主进程
    @abstractmethod
    def stop(self) : pass

    # 获得状态信息
    @abstractmethod
    def getState(self) -> str : pass

    # 检查主进程是否正在运行
    @abstractmethod
    def isRunning(self) -> bool : pass

    @abstractmethod
    def isNormal(self) -> bool : pass

    # 获得样本的统计信息，以字典的方式返回file.json的内容
    @abstractmethod
    def getStatistics(self) -> dict : pass

    # 获得样本的总体数量
    @abstractmethod
    def getCount(self) -> int : pass

# 将两个相同结构的字典进行合并
def mergeDictionary(dic1 : dict, dic2 : dict) -> dict :
    if len(dic1) == 0 : return dic2
    if len(dic2) == 0 : return dic1
    for i, j in dic2.items() :
        if type(j) == dict : 
            if i in dic1.keys() : dic1[i] = mergeDictionary(j, dic1[i])
            else : dic1[i] = j
        elif i not in dic1.keys() : dic1[i] = j
        else : dic1[i] += j
    return dic1

# 将字典转换成字符串显示，retract为首行缩进
def showDictionary(dic : dict, retract = 0) -> str :
    result = ""
    for i , j in dic.items() :
        result += "\n"+ '\t'*retract + str(i) + " : "
        if type(j) == dict : result += "\n" + showDictionary(j, retract + 1)
        else : result += str(j)
    return result.strip('\n')

# 获得文件类型
def getFileType(filepath : str) -> str :
    filehead = ""
    with open(filepath, "rb") as f :
        filehead = f.read(12).hex().upper()
    for i in range(24, 3, -1) :
        if filehead[:i] in fileTypeList.keys() : return fileTypeList[filehead[:i]]
    return "others"

# 文件大小分类转化   
def fileSizeTransform(bytesize : int) :
    bytesize /= 1024
    if bytesize < 1 : return "<1KB"
    elif bytesize >= 1 and bytesize < 10 : return "1KB~10KB"
    elif bytesize >= 10 and bytesize < 100 : return "10KB~100KB"
    elif bytesize >= 100 and bytesize < 1024 : return "10KB~1MB"
    elif bytesize >= 1024 and bytesize < 10240 : return "1MB~10MB"
    else : return ">10MB"

# 计算获得文件时间
def getFileTime(filesha256 : str, rootpath : str) -> str:
    path = rootpath + "/" + filesha256[0] + "/" + filesha256[1] + "/" + filesha256[2] + "/" + filesha256[3] + "/" + filesha256 + ".json"
    if not os.path.exists(path) : return None
    time = ""
    time_num = 9000000000000
    with open(path, "r") as f :
        data = json.loads(f.read())
        if "Androzoo" in data.keys() : return data["Androzoo"]["update"]
        elif "VirusShare" in data.keys() : return data["VirusShare"]["update"]
        if "results" in data.keys() and "scans" in data["results"].keys() : data = data["results"]["scans"]
        else : return None
        for _, value in data.items() :
            if type(value) == dict and 'update' in value.keys() and time_num > int(value['update']) : 
                time = value['update']
                time_num = int(time)
    if time == "" : return None
    else : return time[:4]

# 传送脚本状态
def sendWebState(scriptname : str, state : int) :
    if state == 0 : state = "stop"
    elif state == 1 : state = "running"
    else : state = "sleeping"
    cont = {}
    cont[scriptname] = state
    with open("./webdata/" + scriptname + ".json", "w") as f :
            cont = f.write(json.dumps({"ScriptState" : cont}))

# 直接利用病毒样本获得IAT
def getVirusIAT(path) -> list :
    if not os.path.exists(path) or os.path.isdir(path) or getFileType(path) != "PE" : return []
    pe = pefile.PE(path)
    result = []
    for iid in pe.DIRECTORY_ENTRY_IMPORT :
        for api in iid.imports :
            if api.name != None and api.name.decode("ascii") in virusPossibleAPI.keys() : 
                result.append(api.name.decode("ascii"))
    return result

# 将独热编码转换成API编码
def onehotToInts(onehot : bytearray) -> list:
    result = []
    for i, value in enumerate(onehot) :
        for pos in range(8) :
            if value & (128 >> pos) : result.append(i * 8 + pos)
    return result

# 将API编码转黄成独热编码
def intsToOnehot(ints : list, width : int) -> bytearray :
    result = bytearray(math.ceil(width / 8))
    for num in ints :
        if num >= width : continue
        result[num // 8] |= 128 >> (num % 8)
    return result

# 将API编码转换成API名称
def intsToAPI(ints : list) -> list :
    apis = list(virusPossibleAPI.keys())
    codes = list(virusPossibleAPI.values())
    result = []
    for temp in ints :
        if temp in codes :
            result.append(apis[codes.index(temp)])
    return result

# 将API名称转换成相应的API编码
def APIToInts(apis : list) -> list :
    result = []
    for temp in apis :
        if temp in virusPossibleAPI.keys() :
            result.append(virusPossibleAPI[temp])
    return result

# 获得病毒API独热编码
def getVirusAPIOnehotBase(sha256 : str, path : str) -> bytearray :
    if not os.path.exists(path) or os.path.isdir(path) : return None
    with open(path, "rb") as f :
        # 读取长度
        onehotlength = int(f.read(4).hex(), 16)
        # 寻找
        a = f.read(32 + onehotlength)
        while len(a) != 0 :
            if a[:32].hex() == sha256 : return bytearray(a[32:])
            a = f.read(32 + onehotlength)
    return None

def getVirusAPIOnehot(sha256 : str, rootpath : str) -> bytearray :
    path = rootpath + "/" + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/APITotal.apit"
    return getVirusAPIOnehotBase(sha256, path)

# 通过APITotal.apit获得病毒API
def getVirusAPI(sha256 : str, rootpath : str) -> list :
    onehot = getVirusAPIOnehot(sha256, rootpath)
    if onehot == None : return None
    else : return intsToAPI(onehotToInts(onehot))

# 写入APITotal.apit
def writeAPITotal(sha256 : str, onehot : bytearray, path : str) :
    buf = None
    # 读取原文件
    if os.path.exists(path) :
        with open(path, "rb") as f :
            buf = bytearray(f.read())
    flag = False
    # 如果原文件不存在或者不完整
    if buf == None or len(buf) < 4 : 
        if len(onehot) > 2 ** 32 : raise Exception("Onehot is too long")
        # 写入长度信息
        buf = bytearray(struct.pack(">i", len(onehot)))
    else :
        if len(onehot) != int(buf[:4].hex(), 16) : raise Exception("Onehot length doesn't match ")
        for i in range(4, len(buf), 32 + len(onehot)) :
            if buf[i : i + 32].hex() == sha256 : 
                for t in range(len(onehot)) : buf[t + i + 32] = onehot[t]
                flag = True
                break
    # 写入
    if not flag : buf += bytearray([int(sha256[t * 2 : t * 2 + 2], 16) for t in range(len(sha256) // 2)]) + onehot
    with open(path, "wb") as f :
        f.write(buf)
        f.flush()
    

# 将API转换成独热编码并写入APITotal.apit
def writeVirusAPIOnehot(sha256 : str, apis : list, rootpath : str) :
    path = rootpath + "/" + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/APITotal.apit"
    writeAPITotal(sha256, intsToOnehot(APIToInts(apis), len(virusPossibleAPI)), path)
    
# 将病毒API信息写入对应的APITotal.apit
def writeVirusAPI(sha256 : str, rootpath : str) :
    path = rootpath + "/" + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3]
    a = getVirusIAT(path + "/" + sha256 + ".az")
    b = getVirusIAT(path + "/" + sha256 + ".vs")
    if len(a) == 0 and len(b) == 0 : return
    elif len(a) > 0 :
        writeVirusAPIOnehot(sha256, a, rootpath)
    else :
        writeVirusAPIOnehot(sha256, b, rootpath)
