import re,json,os,math
class FamilySearch():
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
        searchStr = str(searchStr)
        # 结果初始化
        result = {"Pages" : 0, "Result" : []} 

        # 寻找
        # 获得的是根目录下的file.json
        for repofolder in self.repofolders :
            # 检查file.json文件是否存在
            if not os.path.exists(repofolder + "/file.json") : continue
            # 打开文件
            with open(repofolder + "/file.json", "r") as f :
                info = json.loads(f.read())
                # 检查是否进行了文件家族分布统计
                if not "FamilyDistribution" in info.keys() : continue
                # 正则表达式规则
                ref = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*\.){2}' + searchStr + r"(\.[a-zA-Z0-9-]*)*"
                # 检查是否有目标家族要求的文件
                hasfList=[]
                for keys in info["FamilyDistribution"].keys() : 
                    if re.search(ref,keys) : hasfList.append(keys)

                # 统计总体需要显示的文件数量
                tempnum=0
                if len(hasfList)>0:
                    for keys in hasfList:
                        result["Pages"] += info["FamilyDistribution"][keys]
                        tempnum += info["FamilyDistribution"][keys]
                # 计算位置
                if pos < tempnum :
                    # 检查result有没有满
                    if offset > 0 :
                        res = self.__getList(pos, offset, repofolder, searchStr, 4, repofolder)
                        result["Result"] += res
                        offset -= len(res)
                        pos = 0
                # 跳过
                else : pos -= tempnum
        # 计算显示页数
        result["Pages"] =  math.ceil(result["Pages"] / self.__singlepagenum)
        return result

                    
        
    def __getList(self, pos, offset, path, target, depth, rootpath) :
        # 读取文件列表并排序
        ref = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*\.){2}' + target + r"(\.[a-zA-Z0-9-]*)*"
        directorylist = os.listdir(path)
        directorylist.sort()
        result = []
        # 如果已经到达了底层目录
        if depth <= 0 : 
            for fil in directorylist :
                if fil.split(".")[-1] == 'feat':
                    # 判断文件的家族信息
                    if pos > 0 : 
                        pos -= 1
                        continue
                    with open(path+"/"+fil) as f:
                        fam=json.loads(f.read())["Family"]
                        if re.search(ref,fam):result.append(fil.split(".")[0])
                        # 检查是否满了
                        if len(result) >= offset : break
            return result


        # 如果没有到达根目录
        for folder in directorylist :
            # 检查file.json是否存在
            if not os.path.exists(path + "/" + folder + "/file.json") : continue
            with open(path + "/" + folder + "/file.json", "r") as f :
                info = json.loads(f.read())
                # 检查是否有家族信息，或者有的家族信息是否符合查询条件
                if not "FamilyDistribution" in info.keys() : continue
                familynames=info["FamilyDistribution"].keys()
                haslist=[]
                tempnum=0
                for keys in familynames:
                    if re.search(ref,keys): 
                        haslist.append(keys)
                        tempnum+=info["FamilyDistribution"][keys]
                if len(haslist)==0 : continue
                # 寻找位置
                if pos >= tempnum : 
                    pos -= tempnum
                    continue
                # 查找文件
                res = self.__getList(pos, offset, path + '/' + folder, target, depth - 1, rootpath)
                offset -= len(res)
                result += res
                pos = 0
                # 检查是否满了
                if offset <= 0 : return result
        return result