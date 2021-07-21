import re
from backstage.TimeSearch import TimeSearch
from backstage.FamilySearch import FamilySearch

'''
本脚本为综合之前所有搜索功能的脚本，将两者综合运行
问题在于这个不能直接先显示有多少页（因为需要求几者的交集，要求交集就要先把原本集合全部求出来才知道长度，会降低查询速度）
建议本页面回显结果时只有上下页，记录下他当前在多少页，但是不显示总共有多少页（只能知道最多有多少页，但是不知道详细页数）

问题：当数量比较多的时候查询会快，但是数量较少时，会需要很长时间去查询
'''

class HybirdSearch():
    def __init__(self) -> None:
        self.__IsTime=False
        self.__IsFamily=False
        self.__compreResult=[]
        self.__searchStrList=[]
        
        self.__singePageNum=50

        # 以下变量只有在复合查询的时候才会用到
        # 正常应该显示的页面数量
        self.__showPage=0
        # time里已经查询到的页面number
        self.__timePage=1
        # family里已经查询到的页面number
        self.__familyPage=1

        self.__laststr=""

        # 记录上次剩下没有弄完的，0是两个都没有剩余；1是time有剩余；2是family有剩余
        self.__lastTF=0
        self.__lastList=[]

        self.__allGet=False
        self.__IsTimeAllGet=False
        self.__IsFamilyAllGet=False
    
    def selectSearchCondition(self,conditionList):
        # 选择需要综合使用的查询条件
        # 参数说明：conditionList里存放的是复选状态下对每个搜索选项是否选择，顺序为：[time,family,...]，其中每一项都是一个bool值
        self.__IsTime=conditionList[0]
        self.__IsFamily=conditionList[1]

    def reset(self):
        self.__IsTime=False
        self.__IsFamily=False
        self.__compreResult=[]
        self.__searchStrList=[]

        # 以下变量只有在复合查询的时候才会用到
        # 正常应该显示的页面数量
        self.__showPage = 0
        # time里已经查询到的页面number
        self.__timePage = 1
        # family里已经查询到的页面number
        self.__familyPage = 1

        self.__laststr = ""

        # 记录上次剩下没有弄完的，0是两个都没有剩余；1是time有剩余；2是family有剩余
        self.__lastTF = 0
        self.__lastList = []

        self.__allGet = False
        self.__IsTimeAllGet = False
        self.__IsFamilyAllGet = False

    def getResult(self,searchStr,page):
        # 综合查询
        # 参数说明：searchStrList同样是复选状态下的每个搜索项，如果有某一项没有选，那一项设置成None就好，形式为：["time","family",...]，顺序同上面的conditionList
        timeResult=None
        familyResult=None
        tempResult=[]
        page=int(page)
        self.__showPage=page
        if self.__laststr=="":self.__laststr=searchStr
        if self.__laststr!=searchStr:
            self.reset()
            self.__laststr=searchStr

        # 1. 正则表达式进行判断，将其按照规定顺序排列
        reFamily=r'[a-zA-Z]{1,}'
        reTime=r"[0-9]{4}"
        tempSearchList=searchStr.split()
        fstr=""
        tstr=""
        for i in tempSearchList:
            if re.search(reTime,i):tstr=i
            if re.search(reFamily,i):fstr=i
        self.__searchStrList.append(tstr)
        self.__searchStrList.append(fstr)
        # print(self.__searchStrList)

        # 2. 根据已经排好顺序的序列进行查找，现在要查找的序列是self.__searchStrList
        # 修改查询状态
        if fstr=="":self.__IsFamily=False
        else : self.__IsFamily=True
        if tstr!="":self.__IsTime=True
        else : self.__IsTime=False
        # print("(self.__IsTime)"+str(self.__IsTime))
        # print("(self.__IsFamily)"+str(self.__IsFamily))

        # 开始查询
        if self.__IsTime and not self.__IsFamily:
            # 第一种情况：只用time查询
            t=TimeSearch()
            t.changeSinglePagenum(self.__singePageNum)
            timeResult=t.getResult(self.__searchStrList[0],page)
            return timeResult
        elif not self.__IsTime and self.__IsFamily:
            # 第二种情况：只用Family查询
            f=FamilySearch()
            f.changeSinglePagenum(self.__singePageNum)
            familyResult=f.getResult(self.__searchStrList[1],page)
            return familyResult
        else:
            # print("两个都查")
            # 第三种情况：两个都使用进行查询的


            if page <= len(self.__compreResult):
                ret={}
                ret["Result"]=self.__compreResult[page-1]
                return ret
            if self.__allGet:
                ret={}
                ret["Result"]=[]
                return ret

            currentTimeResult=[]
            currentFamilyResult=[]
            while len(self.__compreResult)<page:
                if self.__allGet:break
                # 选择两者都有的，加入到compreResult中
                while len(tempResult)<self.__singePageNum:
                    t=TimeSearch()
                    t.changeSinglePagenum(self.__singePageNum)
                    f=FamilySearch()
                    f.changeSinglePagenum(self.__singePageNum)

                    # 先和上次剩下的进行一个对比
                    if self.__lastTF==1:
                        c=f.getResult(self.__searchStrList[1],self.__familyPage)
                        currentFamilyResult=c["Result"]
                        if self.__familyPage>c["Pages"]:
                            self.__allGet=True
                            break
                        while len(self.__lastList)>0 and len(currentFamilyResult)>0:
                            if self.__lastList[0]<currentFamilyResult[0]:self.__lastList.pop(0)
                            elif self.__lastList[0]>currentFamilyResult[0] : currentFamilyResult.pop(0)
                            else : 
                                tempResult.append(self.__lastList.pop(0))
                                currentFamilyResult.pop(0)
                        if len(self.__lastList)==len(currentFamilyResult)==0:
                            self.__lastTF=0
                        if len(self.__lastList)>0:
                            self.__familyPage+=1
                        if len(currentFamilyResult)>0:
                            self.__lastTF=2
                            self.__lastList=currentFamilyResult
                    elif self.__lastTF==2:
                        c=t.getResult(self.__searchStrList[0],self.__timePage)
                        currentTimeResult=c["Result"]  
                        if self.__timePage>c["Pages"]:
                            self.__allGet=True
                            break
                        while len(self.__lastList)>0 and len(currentTimeResult)>0:
                            if self.__lastList[0]<currentTimeResult[0]:self.__lastList.pop(0)
                            elif self.__lastList[0]>currentTimeResult[0] : currentTimeResult.pop(0)
                            else : 
                                tempResult.append(self.__lastList.pop(0))
                                currentTimeResult.pop(0)
                        if len(self.__lastList)==len(currentTimeResult)==0:
                            self.__lastTF=0
                        if len(self.__lastList)>0:
                            self.__timePage+=1
                        if len(currentTimeResult)>0:
                            self.__lastTF=1
                            self.__lastList=currentTimeResult
                    else:
                        tc=t.getResult(self.__searchStrList[0],self.__timePage)
                        tf=f.getResult(self.__searchStrList[1],self.__familyPage)
                        currentTimeResult=tc["Result"]
                        # print("cT")
                        # print(tc["Pages"])
                        # print(currentTimeResult)
                        currentFamilyResult=tf["Result"]
                        # print("cF")
                        # print(tf["Pages"])
                        # print(currentFamilyResult)
                        # print()
                        # print()
                        if self.__familyPage>tf["Pages"] or self.__timePage>tc["Pages"]:
                            self.__allGet=True
                            break
                        # 加入两者都有的，
                        while len(currentTimeResult)>0 and len(currentFamilyResult)>0:
                            if currentTimeResult[0]<currentFamilyResult[0]:currentTimeResult.pop(0)
                            elif currentTimeResult[0]>currentFamilyResult[0]:currentFamilyResult.pop(0)
                            else : 
                                tempResult.append(currentFamilyResult.pop(0))
                                currentTimeResult.pop(0)
                        # print("pop之后")
                        # print(currentTimeResult)
                        # print(currentFamilyResult)
                        if len(currentTimeResult)==0 and len(currentFamilyResult)==0:
                            self.__allGet=True
                            break
                        # 下次循环的时候要翻页了
                        if len(currentTimeResult)==0:
                            self.__timePage+=1
                        if len(currentFamilyResult)==0: 
                            self.__familyPage+=1
                        # 把本次没有扫完的存起来，方便下次循环查
                        if len(currentTimeResult)>0:
                            self.__lastList=currentTimeResult
                            self.__lastTF=1
                        if len(currentFamilyResult)>0:
                            self.__lastTF=2
                            self.__lastList=currentFamilyResult
                # print(tempResult)
                self.__compreResult.append(tempResult)
            ret={}
            ret["Result"]=self.__compreResult[page-1]
            return ret

    def changeSinglePage(self,newSinge):
        self.__singePageNum=newSinge