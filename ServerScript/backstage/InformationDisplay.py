import flask
import json
import os

class InformationDisplay :
    def __init__(self) :
        # 数据来源列表
        self.files = ["webdata/nkaz.json", "webdata/nkvt.json", "webdata/nkvs.json", "webdata/historydata.json", "../NKVSDATA/data.json", "../NKAZDATA/file.json"]
        
        self.totalFileNumber = 0
        self.fileTimeDistribution = []
        self.fileSizeDistribution = []
        self.fileTypeDistribution = []
        self.scriptState = []
        self.totalSampleData = []
        self.familyDistribution = []
        self.__fileTimeDistributionMap = {}
        self.__fileSizeDistributionMap = {}
        self.__fileTypeDistributionMap = {}
        self.__scriptStateMap = {}
        self.__totalSampleDataMap = {}
        self.__familyDistributionMap = {}
        self.updateData()
    
    # 更新数据
    def updateData(self) :
        self.reset()
        for file in self.files :
            if os.path.exists(file) :
                with open(file) as f_obj :
                    data = json.load(f_obj)
                    if "TotalFileNumber" in data.keys() :
                        self.totalFileNumber += data["TotalFileNumber"]

                    # 时间分布
                    if "FileTimeDistribution" in data.keys() : 
                        for temp in data["FileTimeDistribution"].items() :
                            if temp[0] in self.__fileTimeDistributionMap.keys() :
                                self.__fileTimeDistributionMap[temp[0]] += temp[1]
                            else :
                                self.__fileTimeDistributionMap[temp[0]] = temp[1]

                    # 大小分布
                    if "FileSizeDistribution" in data.keys() : 
                        for temp in data["FileSizeDistribution"].items() :
                            if temp[0] in self.__fileSizeDistributionMap.keys() :
                                self.__fileSizeDistributionMap[temp[0]] += temp[1]
                            else :
                                self.__fileSizeDistributionMap[temp[0]] = temp[1]
                        
                    # 种类分布
                    if "FamilyDistribution" in data.keys() : 
                        for temp in data["FamilyDistribution"].items() : 
                            familywhole = temp[0].split('.')
                            if len(familywhole) >= 2 : familywhole = familywhole[1]
                            else : continue
                            if familywhole in self.__fileTypeDistributionMap.keys() :
                                self.__fileTypeDistributionMap[familywhole] += temp[1]
                            else :
                                self.__fileTypeDistributionMap[familywhole] = temp[1]

                    # 脚本状态
                    if "FamilyDistribution" in data.keys() : 
                        for temp in data["FamilyDistribution"].items() : 
                            familywhole = temp[0].split('.')[0]
                            if familywhole in self.__scriptStateMap.keys() :
                                self.__scriptStateMap[familywhole] += temp[1]
                            else :
                                self.__scriptStateMap[familywhole] = temp[1]

                    # 样本库数据变化状态
                    if "TotalSampleData" in data.keys() : 
                        for temp in data["TotalSampleData"].items() :
                            if temp[0] in self.__totalSampleDataMap.keys() :
                                self.__totalSampleDataMap[temp[0]] += temp[1]
                            else :
                                self.__totalSampleDataMap[temp[0]] = temp[1]

                    # 病毒家族信息
                    if "FamilyDistribution" in data.keys() : 
                        for temp in data["FamilyDistribution"].items() : 
                            familywhole = temp[0].split('.')
                            if len(familywhole) >= 3 : familywhole = familywhole[2]
                            else : continue
                            if familywhole in self.__familyDistributionMap.keys() :
                                self.__familyDistributionMap[familywhole] += temp[1]
                            else :
                                self.__familyDistributionMap[familywhole] = temp[1]

        # 数据格式转化
        self.fileTimeDistribution = [[temp[0], temp[1]] for temp in self.__fileTimeDistributionMap.items()]
        self.fileSizeDistribution = [[temp[0], temp[1]] for temp in self.__fileSizeDistributionMap.items()]
        self.fileTypeDistribution = [[temp[0], temp[1]] for temp in self.__fileTypeDistributionMap.items()]
        self.scriptState = [[temp[0], temp[1]] for temp in self.__scriptStateMap.items()]
        self.totalSampleData = [[temp[0], temp[1]] for temp in self.__totalSampleDataMap.items()]
        self.familyDistribution = [[temp[0], temp[1]] for temp in self.__familyDistributionMap.items()]

        self.fileTimeDistribution.sort(key = lambda x : x[1], reverse = True)
        self.fileSizeDistribution.sort(key = lambda x : x[1], reverse = True)
        self.fileTypeDistribution.sort(key = lambda x : x[1], reverse = True)
        self.familyDistribution.sort(key = lambda x : x[1], reverse = True)
        self.scriptState.sort(key = lambda x : x[1], reverse = True)
        
        self.fileTimeDistribution = self.fileTimeDistribution[ : 10]
        self.fileSizeDistribution = self.fileSizeDistribution[ : 10]
        self.fileTypeDistribution = self.fileTypeDistribution[ : 10]
        self.familyDistribution = self.familyDistribution[ : 5]

        self.fileTimeDistribution.sort(key = lambda x : x[0])
        self.fileSizeDistribution.sort(key = lambda x : x[0])
        self.fileTypeDistribution.sort(key = lambda x : x[0])
        self.familyDistribution.sort(key = lambda x : x[0])

    # 用于重置数据
    def reset(self) :
        self.totalFileNumber = 0
        self.fileTimeDistribution = []
        self.fileSizeDistribution = []
        self.fileTypeDistribution = []
        self.scriptState = []
        self.totalSampleData = []
        self.familyDistribution = []
        self.__fileTimeDistributionMap = {}
        self.__fileSizeDistributionMap = {}
        self.__fileTypeDistributionMap = {}
        self.__scriptStateMap = {}
        self.__totalSampleDataMap = {}
        self.__familyDistributionMap = {}