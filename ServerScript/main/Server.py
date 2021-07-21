import hashlib
import getpass
import rsa
import socket
from time import sleep

# 0 身份验证
AUTHENTICATION = "0"
# 1 请求指令
INSTRUCTION = "1"
# 2 显示信息
MESSAGESHOW = "2"
# 3 行为确认
INSTCONFIRM = "3"
# 4 获得密码
PASSWORDGET = "4"
# 5 关闭终端
CLOSETERMINAL = "5"

class Client :
    def __init__(self, server) :
        self.__server = server

    # 向用户发送信息并显示在用户的终端上
    def sendMessageShow(self, message : str) :
        self.__server.sendMessageShow(message)
    
    # 身份验证，times表示可以最多输入多少次错误密码，验证通过返回True，失败返回False
    def authentication(self, times = 5) :
        return self.__server.authentication(times)
    
    # 行为确认，message为提示信息，confirmway为用户需要输入的字符，确认成功返回True，否则为False
    def instructionConfirm(self, message : str, *confirmway : str) :
        return self.__server.instructionConfirm(message, confirmway)

    # 发送信息并获得用户反馈
    def getInput(self, message = "") :
        if message != None and message != "": self.__server.sendMessageShow(message)
        return self.__server.getMessage()
        

class Server : 
    def __init__(self, port = 4000) :
        self.__port = port
        self.__host = ""
        self.__BUFSIZE = 2048
        self.__passhash = ""
        self.__client = None
        self.__maxClient = 0
        self.__RSASIZE = 1024
        self.__server = None
        self.__oldClient = None
        # 初始化端口服务
        self.start()

    def __messageGenerator(self, typ : str, *content : str) :
        for i in content : typ += ' ' + i
        return bytes(typ, encoding = "utf-8")

    def __getMessage(self) :
        if not self.hasClient() : raise Exception("No Client")
        return self.__client.recv(self.__BUFSIZE).decode('utf-8')

    def getMessage(self) :
        self.__sendMessage(self.__messageGenerator(INSTCONFIRM, ""))
        return self.__getMessage()
    
    def __sendMessage(self, message : bytes) :
        self.__client.send(message)
        sleep(0.1)
    
    def getClientClass(self) :
        return Client(self)

    def setPort(self, port : int) :
        if self.isServing() : raise Exception("Server is running. You cannot set port.")
        self.__port = port
    
    def getPort(self) :
        return self.__port

    def waitClient(self) :
        # if self.__client == None : 
        self.__client, addr = self.__server.accept()
        print(str(addr))
    
    def hasClient(self) :
        return self.__server != None and self.__client != None

    def isServing(self) :
        return self.__server != None

    def closeClient(self) :
        if self.__client != None :
            self.__client.close()
            self.__oldClient = self.__client
            self.__client = None
    
    def close(self) :
        if self.__server != None :
            self.closeClient()
            self.__server.close()
            self.__server = None

    def setPassword(self, passwd : str) :
        self.__passhash = hashlib.sha256(passwd.encode("utf-8")).hexdigest()

    def start(self) :
        if self.isServing() : return
        self.__server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.__server.bind((self.__host, self.__port))
        self.__server.listen(self.__maxClient)

    # 等待用户发送指令
    def getInstruction(self) :
        if not self.hasClient() : raise Exception("No Client")
        self.__sendMessage(self.__messageGenerator(INSTRUCTION))
        return self.__getMessage().strip()

    # 向用户发送信息并显示在用户的终端上
    def sendMessageShow(self, message : str) :
        if not self.hasClient() : raise Exception("No Client")
        if message == "" : return
        self.__sendMessage(self.__messageGenerator(MESSAGESHOW, message))
        sleep(0.1)

    # 身份验证，times表示可以最多输入多少次错误密码，验证通过返回True，失败返回False
    def authentication(self, times = 5) :
        if not self.hasClient() : raise Exception("No Client")
        elif self.__passhash == "" : return True

        # 形成RSA公钥和私钥
        (publickey, privatekey) = rsa.newkeys(self.__RSASIZE)
        # 生成信息
        meg = self.__messageGenerator(AUTHENTICATION, str(publickey.n), str(publickey.e))
        # 接受对方密码
        for _ in range(times) :
            self.__sendMessage(meg)
            # 解密并检查密码正确性
            passwd = hashlib.sha256(rsa.decrypt(self.__client.recv(self.__BUFSIZE), privatekey)).hexdigest()
            if passwd == self.__passhash : return True
            self.sendMessageShow("Password wrong. Please try again.")
        # 超过错误次数则停止
        self.sendMessageShow("Too many times wrong.")
        return False
    
    # 行为确认，message为提示信息，确认成功返回True，否则为False
    def instructionConfirm(self, message : str, *confirmway : str) :
        if not self.hasClient() : raise Exception("No Client")

        # 发送信息
        self.__sendMessage(self.__messageGenerator(INSTCONFIRM, message))

        # 信息比对
        if self.__getMessage() in confirmway[0] : return True
        return False

    # 关闭对方脚本
    def closeScript(self) :
        if not self.hasClient() : raise Exception("No Client")
        self.__sendMessage(self.__messageGenerator(CLOSETERMINAL))

    def getClientPassword(self) :
        if not self.hasClient() : raise Exception("No Client")

        # 形成RSA公钥和私钥
        (publickey, privatekey) = rsa.newkeys(self.__RSASIZE)
        # 生成信息
        meg = self.__messageGenerator(PASSWORDGET, str(publickey.n), str(publickey.e))
        # 请求对方密码
        self.__sendMessage(meg)

        # 检验传输正确性
        buf = self.__client.recv(self.__BUFSIZE)
        sha = buf[:32]
        buf = buf[32:]
        if hashlib.sha256(buf).digest() != sha : return ""
        
        # 解密返回密码
        return rsa.decrypt(buf, privatekey).decode("utf-8")
        
