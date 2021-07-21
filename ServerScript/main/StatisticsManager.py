import sys
import os
import json
from main.Logger import Logger
from main.Models import *
import time

# 向file.json中添加病毒信息
def addFileJson(information : list, rootpath) :
    for info in information :
        # 加载信息
        filejson = "/file.json"
        filename = info
        featname = filename + ".feat"
        # 生成目录路径
        path = rootpath + "/" + filename[0] + "/" + filename[1] + "/" + filename[2] + "/" + filename[3]

        # if os.path.exists(path + "/" + filename + ".az") : filename = filename + ".az"
        # elif os.path.exists(path + "/" + filename + ".vs") : filename = filename + ".vs"
        # else : continue
        if not os.path.exists(path + "/" + filename) : continue

        # 检查路径
        if not os.path.exists(path) : continue
        if not os.path.exists(path + "/" + filename) : continue
        if not os.path.exists(path + "/" + featname) : continue

        # 获得文件信息
        filesize = fileSizeTransform(os.path.getsize(path + "/" + filename))
        filetype = getFileType(path + "/" + filename)
        filetime = getFileTime(info, rootpath)
        if filetime == None : filetime = "Other"
        
        # 获得家族信息
        filefamily = ""
        with open(path + '/' + featname, "r") as f:
            filefamily = json.loads(f.read())["Family"]
             

        # 生成文件路径
        path += filejson
        cont = ""
        temp = {}
        if os.path.exists(path) :
            with open(path, "r") as f :
                cont = f.read()
                if cont == "" : temp = {}
                else : temp = json.loads(cont)

        with open(path, "w") as f :
            # 读取并更新文件

            # 时间
            if "FileTimeDistribution" not in temp.keys() : 
                temp["FileTimeDistribution"] = {}
                temp["FileTimeDistribution"][filetime] = 1
            else : 
                if filetime not in temp["FileTimeDistribution"].keys() : temp["FileSizeDistribution"][filetime] = 1
                else : temp["FileTimeDistribution"][filetime] += 1

            # 大小
            if "FileSizeDistribution" not in temp.keys() : 
                temp["FileSizeDistribution"] = {}
                temp["FileSizeDistribution"][filesize] = 1
            else : 
                if filesize not in temp["FileSizeDistribution"].keys() : temp["FileSizeDistribution"][filesize] = 1
                else : temp["FileSizeDistribution"][filesize] += 1
            
            # 种类
            if "FileTypeDistribution" not in temp.keys() : 
                temp["FileTypeDistribution"] = {}
                temp["FileTypeDistribution"][filetype] = 1
            else : 
                if filetype not in temp["FileTypeDistribution"].keys() : temp["FileTypeDistribution"][filetype] = 1
                else : temp["FileTypeDistribution"][filetype] += 1

            # 家族
            if "FamilyDistribution" not in temp.keys() : 
                temp["FamilyDistribution"] = {}
                temp["FamilyDistribution"][filefamily] = 1
            else : 
                if filefamily not in temp["FamilyDistribution"].keys() : temp["FamilyDistribution"][filefamily] = 1
                else : temp["FamilyDistribution"][filefamily] += 1

            # 总数
            if "TotalFileNumber" not in temp.keys() : 
                temp["TotalFileNumber"] = 1
            else : temp["TotalFileNumber"] += 1
            
            # 写回
            f.write(json.dumps(temp))
            f.flush()
    

def updateTree(dirtylist : list, rootpath) :
    filejson = "/file.json"
    if not rootpath == getNKAZPath() and not rootpath == getNKVSPath() : return

    # 更新底层目录
    addFileJson(dirtylist, rootpath)


    for h in range(4) :
        # 提取出脏文件
        dirtytemp = set()
        for temp in dirtylist : dirtytemp.add(temp[ : 3 - h])
        dirtylist = dirtytemp
        
        # 更新脏文件
        for temp in dirtylist :
            path = rootpath
            for i in range(len(temp)) : path += "/" + temp[i]
            res = {}
            # 读取数据
            for i in range(16) : 
                subfile = path + "/" + str(hex(i)).lower()[-1]
                if not os.path.exists(subfile) : os.makedirs(subfile)
                subfile += filejson
                if not os.path.exists(subfile) : continue
                with open(subfile, "r") as f :
                    res = mergeDictionary(res, json.loads(f.read()))

            if not os.path.exists(path) : os.makedirs(path)
            path += filejson
            with open(path, "w") as f :
                f.write(json.dumps(res))
                f.flush()


def checkDatabaseBase(rootpath : str, statevalue) :
    folders = "0123456789abcdef"
    dirtylist = []
    for folder1 in folders :
        for folder2 in folders :
            for folder3 in folders :
                for folder4 in folders :
                    statevalue.value += 1
                    # 生成路径
                    path = rootpath + "/" + folder1 + '/' + folder2 + "/" + folder3 + "/" + folder4
                    if not os.path.exists(path) : continue
                    dirtylist.append(folder1 + folder2 + folder3 + folder4)
                    result = {
                        "TotalFileNumber" : 0,
                        "FileTimeDistribution" : {},
                        "FileSizeDistribution" : {},
                        "FileTypeDistribution" : {},
                        "FamilyDistribution" : {}
                    }
                    for fil in os.listdir(path) :
                        if fil.split(".")[-1] == "az" or fil.split(".")[-1] == "vs" :
                            sha256 = fil.split(".")[0]
                            featname = sha256 + ".feat"
                            samplejson = sha256 + ".json"
                            
                            # 检查标签完整性
                            if not os.path.exists(path + "/" + featname) or not os.path.exists(path + "/" + samplejson) :
                                with open(rootpath + "/dirtyfile.ckp.temp", "a") as f :
                                    f.write(sha256 + "\n")
                                continue
                            filetime = ""
                            filetype = ""
                            filesize = ""
                            filefamily = ""
                            try :
                                filetime = getFileTime(sha256, rootpath)
                                if filetime == None : filetime = "Other"
                                filetype = getFileType(path + '/' + fil)
                                filesize = fileSizeTransform(os.path.getsize(path + "/" + fil))
                                with open(path + '/' + featname, "r") as f:
                                    data = json.loads(f.read())
                                    if "Family" not in data.keys() : continue
                                    filefamily = data["Family"]
                            except Exception as e :
                                continue

                            # 总数统计
                            result["TotalFileNumber"] += 1
                            # 时间统计
        
                            if filetime not in result["FileTimeDistribution"].keys() : result["FileTimeDistribution"][filetime] = 1
                            else : result["FileTimeDistribution"][filetime] += 1
                            # 类型统计
                            if filetype not in result["FileTypeDistribution"].keys() : result["FileTypeDistribution"][filetype] = 1
                            else : result["FileTypeDistribution"][filetype] += 1
                            # 大小统计
                            if filesize not in result["FileSizeDistribution"].keys() : result["FileSizeDistribution"][filesize] = 1
                            else : result["FileSizeDistribution"][filesize] += 1
                            # 家族统计
                            if filefamily not in result["FamilyDistribution"].keys() : result["FamilyDistribution"][filefamily] = 1
                            else : result["FamilyDistribution"][filefamily] += 1
                    with open(path + "/file.json", "w") as f :
                        f.write(json.dumps(result))
        # 转移dirtyfile
        if not os.path.exists(rootpath + "/dirtyfile.ckp.temp") : continue
        if not os.path.exists(rootpath + '/dirtyfile') : os.makedirs(rootpath + '/dirtyfile')
        os.popen("mv " + rootpath + "/dirtyfile.ckp.temp " + rootpath + "/dirtyfile/" + str(int(time.time())) + ".ckp")
    # 更新树
    updateTree(dirtylist, rootpath)

def checkDatabase(rootpaths : list, statevalue) :
    # 重置
    statevalue.value = -1
    for path in rootpaths :
        checkDatabaseBase(path, statevalue)