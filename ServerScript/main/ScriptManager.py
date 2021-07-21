from nkaz.nkaz import Androzoo
from nkvs.nkvs import VirusShare
from nkvt.nkvt import VirusTotal
# from main.Server import Server
from main.Server import Client
from main.InstManager import InstManager
from main.ExceptionManager import ExceptionManager
from main.StatisticsManager import checkDatabase
from time import sleep
from main.Models import *
import multiprocessing
import json
import time

class ScriptManager :
    def __init__(self, password = "123456", port = 4000, debug = "") :
        # 脚本初始化
        self.__instManager = InstManager()
        self.__expmanager = ExceptionManager(self.__instManager)
        self.__androzoo = None
        self.__virusShare = None
        self.__virusTotal = None
        self.__nkvspath = getNKVSPath()
        self.__nkazpath = getNKAZPath()
        self.__checkdbstate = None
        self.__checkdbproc = None
        self.__updateproc = multiprocessing.Process(target = self.updateWebdata)
        # 调试模式检查
        port = port + 1
        if debug == "nkaz" : self.__androzoo = Androzoo(self.__instManager, self.__nkazpath)
        elif debug == "nkvs" : self.__virusShare = VirusShare(self.__instManager, self.__nkvspath)
        elif debug == "nkvt" : self.__virusTotal = VirusTotal(self.__instManager, self.__nkazpath, self.__nkvspath)
        elif debug == "main" : pass
        else :
            # 非调试模式则改回默认设置
            port = port - 1
            self.__androzoo = Androzoo(self.__instManager, self.__nkazpath)
            self.__virusShare = VirusShare(self.__instManager, self.__nkvspath)
            # self.__virusTotal = VirusTotal(self.__instManager, self.__nkazpath, self.__nkvspath)

        # 服务启动
        # self.__server = Server(port)
        # self.__client = self.__server.getClientClass()
        # self.__server.setPassword(password)
        # self.__updateproc.start()

        # 注册main指令
        self.__instRegister()

    def updateWebdata(self) :
        hasUpdate = False
        while True :
            # 检查是否到更新时间
            if hasUpdate == True and time.localtime().tm_wday != 1 : hasUpdate = False
            if hasUpdate != False or time.localtime().tm_wday != 1 : 
                # 等待
                time.sleep(1)
                continue
            cont = {}
            result = {}
            # 读取原数据
            if os.path.exists("./webdata/historydata.json") : 
                with open("./webdata/historydata.json", "r") as f :
                    temp = json.loads(f.read())
                    if "TotalSampleData" in temp.keys() : cont = temp["TotalSampleData"]
            # 计算现有数据
            count = 0
            if self.__androzoo != None : count += self.__androzoo.getCount()
            if self.__virusShare != None : count += self.__virusShare.getCount()
            for i in range(8, 0, -1) :
                if str(i - 1) + "周前" in cont.keys() : result[str(i) + "周前"] = cont[str(i - 1) + "周前"]
            if "现在" in cont.keys() : result["1周前"] = cont["现在"]
            result["现在"] = count
            # 写入
            with open("./webdata/historydata.json", "w") as f:
                f.write(json.dumps({"TotalSampleData" : result}))
            hasUpdate = True

    # main指令注册
    def __instRegister(self) :
        self.__instManager.addInstruction("stop", targetfunc = self.stop, targetscript = "main", description = "Shut down all scripts", level = 1)
        self.__instManager.addInstruction("restart", targetfunc = self.restart, targetscript = "main", description = "Restart all scripts", level = 1)
        self.__instManager.addInstruction("changepassword", targetfunc = self.changePassword, targetscript = "main", description = "Change Password", level = 0)
        self.__instManager.addInstruction("help", targetfunc = self.insthelp, targetscript = "main", description = "Script Helper", level = 0)
        self.__instManager.addInstruction("count", targetfunc = self.showCount, targetscript = "main", description = "Count the number of the virus in this database", level = 0)
        self.__instManager.addInstruction("info", targetfunc = self.showInfo, targetscript = "main", description = "Do statistics and show copyright", level = 0)
        self.__instManager.addInstruction("state", targetfunc = self.showState, targetscript = "main", description = "show state", level = 0)
        self.__instManager.addInstruction("check", targetfunc = self.checkAll, targetscript = "main", description = "check all script", level = 0)
        self.__instManager.addInstruction("checkdb", targetfunc = self.checkDB, targetscript = "main", description = "check the database", level = 0)

    # 自动运行
    def auto(self) : pass
        # while True :
            # 等待接入
            # print("waiting")
            # self.__server.waitClient()
            # 身份验证
            # if not self.__server.authentication() : 
            #     print("Authentication doesn't pass")
            #     continue

            # 指令处理
            # while True :
                # sleep(0.1)

                # if self.__expmanager.getNewCount() > 0 : 
                #     self.__server.sendMessageShow("New error or exception. Input 'error' to check")

                # 接受指令
                # meg = self.__server.getInstruction()

                # 退出指令
                # if meg == "exit" : 
                #     self.__server.closeScript()
                #     self.__server.closeClient()
                #     break

                # 解析指令
                # inst = self.__instManager.parseInstruction(meg)
                
                # 运行指令
                # if len(inst) == 0 : 
                #     self.__server.sendMessageShow("No such instruction. Input 'help' for help")
                # elif inst[1] >= 1 and self.__server.authentication() : inst[0](self.__client, tuple(inst[2]))
                # elif inst[1] == 0 : inst[0](self.__client, tuple(inst[2]))

    # 获得异常处理器
    def getExceptionManager(self) -> type(ExceptionManager):
        return self.__expmanager

    # 重启所有脚本
    def restart(self, client : type(Client), args : tuple) :
        client.instructionConfirm("WARNING: This instruction will restart three scripts. \nInput Y or y to confirm...[Y/N]", "Y", "y")
        client.sendMessageShow("Restarting...")
        if self.__androzoo != None : 
            self.__androzoo.stop()
            self.__androzoo.start()
        if self.__virusShare != None :
            self.__virusShare.stop()
            self.__virusShare.start()
        if self.__virusTotal != None :
            self.__virusTotal.stop()
            self.__virusTotal.start()
        self.__checkdbproc = None
        client.sendMessageShow("Complete!")
    
    # 停止所有脚本
    def stop(self, client : type(Client), args : tuple) :
        client.instructionConfirm("WARNING: This instruction will shut down all scripts, including the serverscript. \nInput y or Y to confirm. [Y/N]", 'y', 'Y')
        client.sendMessageShow("Shutting down...")
        if self.__androzoo != None : self.__androzoo.stop()
        if self.__virusShare != None : self.__virusShare.stop()
        if self.__virusTotal != None : self.__virusTotal.stop()
        if self.__checkdbproc != None : self.__checkdbproc.kill()
        self.__updateproc.kill()
        self.__server.close()
        exit(0)
    
    # 获得帮助
    def insthelp(self, client : type(Client), args : tuple) :
        client.sendMessageShow(self.__instManager.getHelp())

    # 更改密码
    def changePassword(self, client : type(Client), args : tuple) :
        if not client.instructionConfirm("WARNING: This instruction will change the password. \nInput y or Y to confirm. [Y/N]", 'y', 'Y') : return
        # 身份验证
        client.sendMessageShow("Old password")
        if not client.authentication() : return
        # 输入新密码
        client.sendMessageShow("New password")
        password = self.__server.getClientPassword()
        # 检测输入正确性
        while password == "" :
            client.sendMessageShow("Invalid Password!")
            password = self.__server.getClientPassword()
        # 设置新密码
        self.__server.setPassword(password)

    # 显示计数
    def showCount(self, client : type(Client), args : tuple) :
        az = 0
        vs = 0
        if self.__androzoo != None : az = self.__androzoo.getCount()
        if self.__virusShare != None : vs = self.__virusShare.getCount()
        message = "Total number : " + str(az + vs)
        if self.__androzoo != None : message += "\n\tnkaz : " + str(az)
        else : message += "\n\tnkaz doesn\'t start"
        if self.__virusShare != None : message += "\n\tnkvs : " + str(vs)
        else : message += "\n\tnkvs doesn\'t start"
        client.sendMessageShow(message) 

    # 显示统计信息
    def showInfo(self, client : type(Client), args : tuple) :
        copyr = "Copyright: nkamg.\n\n"
        total = {}
        if self.__virusShare != None : total = self.__virusShare.getStatistics()
        if self.__androzoo != None : total = mergeDictionary(total, self.__androzoo.getStatistics())
        client.sendMessageShow(copyr + showDictionary(total))

    # 显示运行状态
    def showState(self, client : type(Client), args : tuple) :
        message = "State" 
        if self.__androzoo != None : message += "\n\tnkaz : " + self.__androzoo.getState()
        else : message += "\n\tnkaz doesn\'t start"
        if self.__virusShare != None : message += "\n\tnkvs : " + self.__virusShare.getState()
        else : message += "\n\tnkvs doesn\'t start"
        if self.__virusTotal != None : message += "\n\tnkvt : " + self.__virusTotal.getState()
        else : message += "\n\tnkvt doesn\'t start"
        client.sendMessageShow(message)

    # 检查是否正常
    def checkAll(self, client : type(Client), args : tuple) :
        message = ""
        if self.__androzoo != None : message += "" if self.__androzoo.isNormal() else "nkaz may have problems\n"
        else : message += "\n\tnkaz doesn\'t start"
        if self.__virusShare != None : message += "" if self.__virusShare.isNormal() else "nkvs may have problems\n"
        else : message += "\n\tnkvs doesn\'t start"
        if self.__virusTotal != None : message += "" if self.__virusTotal.isNormal() else "nkvt may have problems\n"
        else : message += "\n\tnkvt doesn\'t start"
        
        if self.__expmanager.getNewCount() > 0 : message += "\nNewError: \n" + showDictionary(self.__expmanager.getCount(), 1)
        if message == "" : message = "All right"
        client.sendMessageShow(message)

    def checkDB(self, client : type(Client), args : tuple) :
        # 检查准备状态
        if self.__checkdbproc == None :
            client.sendMessageShow("Starting...")
            self.__checkdbstate = multiprocessing.Value("i", -1)
            # 检查是否是同一路径
            if getNKAZPath() == getNKVSPath() :
                self.__checkdbproc = multiprocessing.Process(target = checkDatabase, args = ([getNKAZPath()], self.__checkdbstate))
            else : 
                self.__checkdbproc = multiprocessing.Process(target = checkDatabase, args = ([getNKAZPath(), getNKVSPath()], self.__checkdbstate))
            self.__checkdbproc.start()
            client.sendMessageShow("You can input checkdb to check the state.")
        # 检查完成
        elif not self.__checkdbproc.is_alive() :
            if (self.__checkdbstate.value + 1) % (16 ** 4) == 0 :
                client.sendMessageShow("Check Complete!")
            else : client.sendMessageShow("Checkdb crashed!")
            # 重置
            if client.instructionConfirm("If you want to reset checkdb engine to prepare for next check, input Y/y to reset. [Y/N]", "Y", "y") :
                self.__checkdbproc = None
        # 检查进行
        else :
            # 计算进度
            client.sendMessageShow("checking now... \nProcess : " + str(int((self.__checkdbstate.value + 1) * 100 / (2 * (16 ** 4) if getNKAZPath() != getNKVSPath() else 16 ** 4))) + "%")
            # 计算正在检查的位置
            hexstr = str(hex(self.__checkdbstate.value % (16 ** 4)))[2:]
            hexstr = "0" * (4 - len(hexstr)) + hexstr
            if self.__checkdbstate.value >= (16 ** 4) :
                client.sendMessageShow("Checking path : " + getNKVSPath() + "/" + hexstr[0] + "/" + hexstr[1] + "/" + hexstr[2] + "/" + hexstr[3])
            else : 
                client.sendMessageShow("Checking path : " + getNKAZPath() + "/" + hexstr[0] + "/" + hexstr[1] + "/" + hexstr[2] + "/" + hexstr[3])
