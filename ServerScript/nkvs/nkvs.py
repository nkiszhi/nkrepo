# 导入网络代理对象
from main.Server import Client
# 导入指令管理器
from main.InstManager import InstManager
# 导入模板
from main.Models import ScriptModel,mergeDictionary,getFileType,sendWebState

from main.StatisticsManager import addFileJson,updateTree

import json,requests,re,os,shutil,threading
import zipfile as zf
from time import sleep
from multiprocessing import Process
from subprocess import Popen,PIPE,DEVNULL


class VirusShare(ScriptModel):
    def __init__(self, instmanager, rootpath):
        # 具体维护脚本初始化程序请放在__init__()中
        self.downloadProcess = None 
        self.manageProcess = None 
        self.__running = 0
        self.__errorStatusCode = [0,0]   # [0] -> login html error    [1] -> download html error
        
        # 文件路径
        self.rootpath = rootpath +'/'
        self.torrentpath = r"nkvs/data/"
        if os.path.exists(self.torrentpath)==False:
            os.mkdir(self.torrentpath)
                
        self.hexstring = '0123456789abcdef'
        self.checkNKVSDATA() # 建立四级目录
        
        #登录信息
        self.username = 'nkamg'
        self.passwd = 'wOIdIqFrPzaI'
        
        #self.JsonUpdateList = []
        
        self.start()
        
        # 注册指令，此处为例子
        # inst表示指令内容，targetfunc表示指令绑定的函数，targetscript表示目标脚本，description表示指令描述，level为指令等级
        instmanager.addInstruction(inst="start",targetfunc = self.startIns, targetscript = "nkvs", description = "nkvs start",level = 1)
        instmanager.addInstruction(inst="stop",targetfunc = self.stopIns, targetscript = "nkvs", description = "nkvs stop",level = 1)
        instmanager.addInstruction(inst="state",targetfunc = self.statusIns, targetscript = "nkvs", description = "show nkvs state",level=0)
        instmanager.addInstruction(inst = "count", targetfunc = self.count, targetscript = "nkvs", description = "nkvs count", level = 0)

    # 此处为例子
    def count(self, client : type(Client), args : tuple) :
        client.sendMessageShow(str(self.getStatistics()["TotalFileNumber"]))

    def startIns(self,client : type(Client), args : tuple):
        status = self.start()
        if status == 0:
            client.sendMessageShow("duplicated start")
        else:
            client.sendMessageShow("nkvs has been started")
        
    def stopIns(self,client : type(Client), args : tuple):
        status = self.stop()
        if status == 0:
            client.sendMessageShow("duplicated stop")
        else:
            client.sendMessageShow("nkvs has been stopped")
    
    def statusIns(self, client : type(Client), args : tuple):
        client.sendMessageShow(self.getState())


    # 启动主进程
    def start(self) : 
        if self.__running ==1:
            return 0
        lock = threading.Lock()
        with lock:
            self.nkvsStart()
            self.__running = 1
            sendWebState("nkvs",1)
        return 1

    # 终止主进程
    def stop(self) : 
        if self.__running ==0:
            return 0
        lock = threading.Lock()
        with lock:
            self.downloadProcess.kill()
            self.manageProcess.kill()
            self.downloadProcess.join()
            self.manageProcess.join()
            os.popen("ps -ef | grep tget | grep -v grep | cut -c 9-15 | xargs kill -9")
            self.__running = 0
            sendWebState("nkvs",0)
        return 1
        


    # 获得状态信息
    def getState(self) -> str : 
        if self.__running==0:
            return "nkvs has been stopped"
        state = os.popen('ps aux|grep tget').read()
        if state.count("tget")>1:
            return "nkvs is running"
        else :
            return "nkvs is sleeping"
        
    # 检查主进程是否正在运行
    def isRunning(self) -> bool : 
        return self.__running

    # 检查主进程的运行是否正常
    def isNormal(self) -> bool : 
        if self.__errorStatusCode==[0,0]:
            return True
        return False

    # 获得样本的统计信息，以字典的方式返回所有file.json的内容汇总
    def getStatistics(self) -> dict : 
#        dic = {}
#        for i in self.hexstring:
#            for j in self.hexstring:
#                for k in self.hexstring:
#                    for l in self.hexstring:
#                        with open(self.rootpath+'/'+i+'/'+j+'/'+k+'/'+l+'/file.json','r') as file:
#                            dic = mergeDictionary(dic,json.load(file))
        dic = {}
        if os.path.exists(self.rootpath+r'/file.json') == False:
            return {}
        with open(self.rootpath+r'/file.json') as file:
            dic = mergeDictionary(dic,json.load(file))
        return dic

    # 获得样本的总体数量
    def getCount(self) -> int :
#        for i in self.hexstring:
#            for j in self.hexstring:
#                for k in self.hexstring:
#                    for l in self.hexstring:
#                        with open(self.rootpath+'/'+i+'/'+j+'/'+k+'/'+l+'/file.json','r') as file:
#                            count = count + json.load(file)["TotalFileNumber"]
        count = 0
        if os.path.exists(self.rootpath+r'/file.json') == False:
            return 0
        with open(self.rootpath+r'/file.json') as file:
            count = json.load(file)["TotalFileNumber"]
        return count





    # 以下是nkvs主程序
    def nkvsStart(self):
        self.downloadProcess = Process(name = "nkvs download", target = self.nkvsStartdownload)
        self.manageProcess = Process(name = "nkvs manage", target = self.nkvsStartmanage)
        
        self.downloadProcess.start()
        self.manageProcess.start()
    
    def nkvsStartdownload(self):
        while True:
            self.getUrl()
            url = self.makesure()
            self.downloadTorrent(url)
            #sleep(7*24*60*60)
            sleep(6*60*60)
                
            
    def nkvsStartmanage(self):
        while True:
            self.unzip()
            self.moveSamples()
            # self.updateJSON()
            sleep(60*60)
    

    
        
    def getUrl(self):
        global html
        # ------------------login html----------------------
        verifyIdentity = {'username': self.username, 'password': self.passwd,}
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/73.0.3683.75 Chrome/73.0.3683.75 Safari/537.36',}
        loginUrl = 'https://virusshare.com/processlogin'
        session = requests.Session()
        loginResp = session.post(url=loginUrl,data=verifyIdentity,headers=headers)
        html_= loginResp.text
        with open(self.torrentpath + "loginHTML.txt","w+") as f:
            f.write(html_)

        if loginResp.status_code != 200:
            self.__errorStatusCode[0]=loginResp.status_code
            return
        else:
            self.__errorStatusCode[0] = 0   # 即使发生过错误但现在能够正常运行
        
        
        # ------------------download html----------------------
        downloadUrl = 'https://virusshare.com/torrents.4n6'
        downloadResp = session.post(url=downloadUrl,headers=headers)
        
        if downloadResp.status_code != 200:
            self.__errorStatusCode[1]=downloadResp.status_code
            return
        else:
            self.__errorStatusCode[1] = 0   # 即使发生过错误但现在能够正常运行

        html = downloadResp.text
        with open(self.torrentpath+"downloadHTML.txt","w+") as f:
            f.write(html)
        
        # ------------------get url----------------------
        urlList = re.findall(r"<a.*?href=\"(.*VirusShare_.*)\">.*<\/a>",html,re.I)
        if not os.path.exists(self.torrentpath + "url_list.txt"):
            with open(self.torrentpath + "url_list.txt", 'w+') as f:
                for item in urlList:
                    f.write("%s\n" % item)
            # return urlList
        else:
            with open(self.torrentpath + "url_list.txt", 'r') as f:   
                old = [line.rstrip('\n') for line in f]
            urlList = set(urlList) - set(old)
            with open(self.torrentpath + "url_list.txt", 'a') as f:    #   全部的url
                for item in urlList:
                    f.write("%s\n" % item)
            # return urlList    #    新增的url
            
            
    def makesure(self):
        url = []
        #for file in os.listdir(self.torrentpath):
        #    if file[-4:] == ".zip" or file[-8:] == '.torrent' :
        #        os.remove(self.torrentpath+file)
        try: 
            with open(self.torrentpath+"url_list.txt",'r') as f :
                totalurl = [ line for line in f.readlines() ]
                for line in totalurl:
                    sign = 0  # 0 表示需要下载
                    for torrent in os.listdir(self.torrentpath+'finishedzip'):
                        if torrent in line:
                            sign = 1 # 不需要下载
                    if sign == 0 :
                        url.append(line)
            return url
        except: # 没有url_list.txt 这个文件 
            return []
        
    def downloadTorrent(self,url):
        url = url[:50]
        for u in url:
            u = u.strip()
            torrentFile = os.path.join(self.torrentpath, u.split('?')[0].split('/')[-1])
            requests.adapters.DEFAULT_RETRIES = 5
            s = requests.session()
            s.keep_alive = False
            requests.packages.urllib3.disable_warnings()
            r = requests.get(u, verify=False)
            with open(torrentFile,'wb') as f:
                f.write(r.content)
            
            # os.system('tget {}'.format(torrentFile))
            with open(self.torrentpath+r"vslog.txt",'a+') as vslog:
                vslog.write('start '+u)
                #print( 'cd '+self.torrentpath+'; '+' tget {} > vslog.txt'.format(u.split('?')[0].split('/')[-1]))
                process = Popen(args = 'cd '+self.torrentpath+'; '+' tget {} > vslog.txt'.format(u.split('?')[0].split('/')[-1]), stderr=vslog,stdout=vslog,shell=True)


    def unzip(self):
        #print("start unzip")
        if os.path.exists(self.torrentpath+r'tempSamples/')==False:
            os.makedirs(self.torrentpath+r'tempSamples/')
        if os.path.exists(self.torrentpath+r'finishedzip')==False:
            os.makedirs(self.torrentpath+r'finishedzip')
        for file in os.listdir(self.torrentpath):
            if file[-4:]=='.zip':
                #print(file)
                try:
                    z = zf.ZipFile(self.torrentpath+file,'r')
                    z.extractall(path = self.torrentpath+r'tempSamples/', pwd="infected", encode = "ascii")
                    shutil.move(self.torrentpath+file,self.torrentpath+r'finishedzip/'+file)
                    shutile.move(self.torrentpath + file[:-4], self.torrentpath+r'finished/' + file[:-4]) # torrent 也移动过去
                except:
                    pass


    def moveSamples(self):
        tempSamplesPath = self.torrentpath+r'tempSamples/'
        files = os.listdir(tempSamplesPath)
        for f in files:
            sha256 = os.popen("sha256sum " + tempSamplesPath+f)
            sha256 = sha256.read().split(' ')[0]
            os.rename(tempSamplesPath+f,tempSamplesPath+sha256+'.vs')
        
        dirtylist = []
        for file in os.listdir(tempSamplesPath):
            if os.path.exists(self.rootpath+'/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file[3]+'/'+file):
                os.remove(tempSamplesPath+file)
                continue
            shutil.move(tempSamplesPath + file, self.rootpath+'/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file[3]+'/'+file)
            dirtylist.append(file[:-3])
            # self.JsonUpdateList.append(file)
            
        if os.path.exists(self.rootpath+'/dirtyfile')==False:
            os.mkdir(self.rootpath+'/dirtyfile')
        
        if dirtylist!= []:
            with open(self.rootpath+'/dirtyfile/' + str(int(time.time()))+'/'+'.vs','w') as f:
                for item in dirtylist:
                    f.write(item+'\n')
        
    def createJSON(self,path):
        if os.path.exists(path)==False:
            jsondict = {}
            jsondict["TotalFileNumber"] = 0
            jsondict["FileTimeDistribution"] = { "2012":0, "2013":0, "2014":0, "2015":0, "2016":0, "2017":0, "2018":0, "2019":0, "2020":0}
            jsondict["FileSizeDistribution"] = { "<10KB":0, "10KB-100KB":0, "100KB-1MB":0, "1MB-10MB":0, ">10MB":0}
            jsondict["FileTypeDistribution"] = { "PEFile":0, "ELF":0, "HTML":0, "Compressed File":0, "other":0}
            f = open(path,'w')
            json.dump(jsondict,f)
            f.close()
    
    # 更新file.json
    def updateJSON(self):
        
        # 如果没有file.json 就新建一个
        self.createJSON(self.rootpath+'/'+"file.json")
        for i in self.hexstring:
            self.createJSON(self.rootpath+'/'+i+'/'+"file.json")
            for j in self.hexstring:
                self.createJSON(self.rootpath+'/'+i+'/'+j+'/'+"file.json")
                for k in self.hexstring:
                    self.createJSON(self.rootpath+'/'+i+'/'+j+'/'+k+'/'+"file.json")
                    for l in self.hexstring:
                        self.createJSON(self.rootpath+'/'+i+'/'+j+'/'+k+'/'+l+'/'+"file.json")
        
        filels = []
        for file in self.JsonUpdateList:
            filedict = {}
            
            #name
            filedict["FileName"] = file
            
            #year
            #info = os.stat(self.rootpath+'/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file[3]+'/'+file)
            #time = info.st_mtime
            #year = time.strftime("%Y",time)
            time = os.path.getmtime(self.rootpath+'/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file[3]+'/'+file)
            filedict["FileTime"] = time
            
            # size
            size = os.path.getsize(self.rootpath+'/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file[3]+'/'+file)
#            if size<=10000: # 10KB
#                filedict["FileSize"] = "<10KB"
#            elif size<=100000: #100KB
#                filedict["FileSize"] = "10KB-100KB"
#            elif size<=10000000: # 1MB
#                filedict["FileSize"] = "100KB-1MB"
#            elif size<=100000000: # 10MB
#                filedict["FileSize"] = "1MB-10MB"
#            else :
#                filedict["FileSize"] = ">10MB"
            filedict["FileSize"] = size
                
            #type
            ty = getFileType(self.rootpath+'/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file[3]+'/'+file)
            filedict["FileType"] = ty
            
            filels.append(filedict)
            
        addFileJson(filels)
        updateTree(self.rootpath)
        
        
    def checkNKVSDATA(self):
        if os.path.exists(self.rootpath)==False:
            os.mkdir(self.rootpath)
        for i in self.hexstring:
            if os.path.exists(self.rootpath+'/'+i)==False:
                os.mkdir(self.rootpath+'/'+i)
            for j in self.hexstring:
                if os.path.exists(self.rootpath+'/'+i+'/'+j)==False:
                    os.mkdir(self.rootpath+'/'+i+'/'+j)
                for k in self.hexstring:
                    if os.path.exists(self.rootpath+'/'+i+'/'+j+'/'+k)==False:
                        os.mkdir(self.rootpath+'/'+i+'/'+j+'/'+k)
                    for l in self.hexstring:
                        if os.path.exists(self.rootpath+'/'+i+'/'+j+'/'+k+'/'+l)==False:
                            os.mkdir(self.rootpath+'/'+i+'/'+j+'/'+k+'/'+l)