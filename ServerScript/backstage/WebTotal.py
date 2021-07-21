import flask
import json
from backstage.InformationDisplay import InformationDisplay
# 在这里修改
from backstage.Sha256Search import Sha256Search
from backstage.TimeSearch import TimeSearch
from backstage.FamilySearch import FamilySearch
from backstage.HybirdSearch import HybirdSearch

import time

class Web :
    def __init__(self) :
        self.informationDisplay = InformationDisplay()
        self.Sha256Search = Sha256Search()
        self.TimeSearch=TimeSearch()
        self.FamilySearch=FamilySearch()
        self.HybirdSearch=HybirdSearch()

    def update(self) :
        self.informationDisplay.updateData()

    def getSha256Result(self, searchContent) :
        return json.dumps(self.Sha256Search.getResult(searchContent))

    def getTimeResult(self, searchContent, page) :
        # 在这里修改
        return json.dumps(self.TimeSearch.getResult(searchContent, page))

    def getFamilySearchResult(self, searchContent, page) :
        return json.dumps(self.FamilySearch.getResult(searchContent, page))
    
    def getHybirdSearchResult(self,searchContent,page):
        return json.dumps(self.HybirdSearch.getResult(searchContent,page))

    def getMainWeb(self) :
        return flask.render_template('index.html', \
            FileTimeDistribution = self.informationDisplay.fileTimeDistribution, \
                FileSizeDistribution = self.informationDisplay.fileSizeDistribution, \
                    FileTypeDistribution = self.informationDisplay.fileTypeDistribution, \
                        TotalFileNumber = self.informationDisplay.totalFileNumber, \
                            ScriptState = self.informationDisplay.scriptState, \
                                TotalSampleData = self.informationDisplay.totalSampleData, \
                                    FamilyDistribution = self.informationDisplay.familyDistribution)
