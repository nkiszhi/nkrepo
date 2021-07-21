import re,json,os,math
from main.Models import getFileTime
class TimeSearch():
    def __init__(self):
        self.repofolders=["../NKAZDATA","../NKVSDATA"]
        self.__singlepagenum = 50
    
    def changeSinglePagenum(self,newsinglenum):
        # 修改每页需要显示的数量
        self.__singlepagenum=newsinglenum

    def getResult(self, searchStr, page) :
        # 计算显示的位置和需要显示的数量
        pos = (int(page) - 1) * self.__singlepagenum
        offset = self.__singlepagenum
        searchStr = str(searchStr)[:4]
        # 结果初始化
        result = {"Pages" : 0, "Result" : []} 
        # 寻找
        for repofolder in self.repofolders :
            # 检查file.json文件是否存在
            if not os.path.exists(repofolder + "/file.json") : continue
            # 打开文件
            with open(repofolder + "/file.json", "r") as f :
                info = json.loads(f.read())
                # 检查是否进行了文件时间分布统计
                if not "FileTimeDistribution" in info.keys() : continue
                # 检查是否有目标时间要求的文件
                if not searchStr in info["FileTimeDistribution"] : continue
                # 统计总体需要显示的文件数量
                result["Pages"] += info["FileTimeDistribution"][searchStr]
                # 计算位置
                if pos < info["FileTimeDistribution"][searchStr] :
                    # 检查result有没有满
                    if offset > 0 :
                        res = self.__getList(pos, offset, repofolder, searchStr, 4, repofolder)
                        result["Result"] += res
                        offset -= len(res)
                        pos = 0
                # 跳过
                else : pos -= info["FileTimeDistribution"][searchStr]
        # 计算显示页数
        result["Pages"] =  math.ceil(result["Pages"] / self.__singlepagenum)
        return result

                    
        
    def __getList(self, pos, offset, path, target, depth, rootpath) :
        # 读取文件列表并排序
        directorylist = os.listdir(path)
        directorylist.sort()
        result = []
        # 如果已经到达了底层目录
        if depth <= 0 : 
            for fil in directorylist :
                # if fil.split(".")[-1] == 'az' or fil.split(".")[-1] == 'vs' :
                if len(fil.split(".")) == 1 :
                    # 判断文件时间
                    if pos > 0 : 
                        pos -= 1
                        continue
                    try :
                        if getFileTime(fil.split(".")[0], rootpath) == target :
                            result.append(fil.split(".")[0])
                            # 检查是否满了
                            if len(result) >= offset : break
                    except Exception as e :
                        continue
            return result
        # 如果没有到达根目录
        for folder in directorylist :
            # 检查file.json是否存在
            if not os.path.exists(path + "/" + folder + "/file.json") : continue
            with open(path + "/" + folder + "/file.json", "r") as f :
                info = json.loads(f.read())
                # 检查是否有目标时间要求的文件和是否对时间分布进行了统计
                if not "FileTimeDistribution" in info.keys() : continue
                if not target in info["FileTimeDistribution"].keys() : continue
                # 寻找位置
                if pos >= info["FileTimeDistribution"][target] : 
                    pos -= info["FileTimeDistribution"][target]
                    continue
                # 查找文件
                res = self.__getList(pos, offset, path + '/' + folder, target, depth - 1, rootpath)
                offset -= len(res)
                result += res
                pos = 0
                # 检查是否满了
                if offset <= 0 : return result
        return result