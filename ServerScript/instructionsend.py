import socket
import getpass
import hashlib
import rsa

# 设置debug模式
debug = False


host = "10.134.78.10" # 目标ip地址，如果是localhost则为本地测试
# host = "10.134.75.233"
port = 4000 # 端口号，默认为4000

# 0 身份验证
AUTHENTICATION = "0"
# 1 请求指令
INSTRUCTION = "1"
# 2 显示信息
MESSAGESHOW = "2"
# 3 行为确认
INSTCONFIRM = "3"
# 4 发送一个密码
PASSWORDSEND = "4"

class Server :
    def __init__(self, host : str, port : int) :
        self.__host = host
        self.__port = port
        self.__BUFSIZE = 2048
        self.__server = None
        self.__DSASIZE = 1024
    
    # 发送密码
    def __sendPassword(self, keymeg : str) :
        if not self.isConnect() : raise Exception("No server.")

        # 获得公钥
        buf_temp = keymeg.split(" ")
        if len(buf_temp) != 2 : 
            print("RSA publickey get failed.")
            return
        publickey = rsa.PublicKey(int(buf_temp[0]), int(buf_temp[1]))

        # 获得密码并加密
        password = rsa.encrypt(bytes(getpass.getpass("password: "), encoding = "utf-8"), publickey)
        # 传送密码
        self.__server.send(password)

    # 传送一个指令
    def __sendInstruction(self, message = "> ") :
        if not self.isConnect() : raise Exception("No server.")
        inst = str(input(message))
        while inst == "" : inst = str(input(message))
        self.__server.send(bytes(inst, encoding = "utf-8"))
    
    # 传送一个密码
    def __sendMyPassword(self, keymeg : str) :
        if not self.isConnect() : raise Exception("No server.")

        # 获得公钥
        buf_temp = keymeg.split(" ")
        if len(buf_temp) != 2 : 
            print("RSA publickey get failed.")
            return
        
        # 获得密码并确认密码，最后进行加密
        publickey = rsa.PublicKey(int(buf_temp[0]), int(buf_temp[1]))
        pas_temp = getpass.getpass("password: ")
        while getpass.getpass("password confirm: ") != pas_temp : pass
        password = rsa.encrypt(bytes(pas_temp, encoding = "utf-8"), publickey)

        # 对密文进行数字签名，然后传输
        self.__server.send(hashlib.sha256(password).digest() + password)

    # 关闭连接
    def close(self) :
        if self.isConnect() :
            self.__server.close()
            self.__server = None

    # 连接
    def connect(self) :
        if not self.isConnect() :
            self.__server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            self.__server.connect((self.__host, self.__port))
    
    # 检测是否连接
    def isConnect(self) :
        if self.__server == None : return False
        return True

    # 设置连接地址
    def setConnect(self, host : str, port : int) :
        if self.isConnect() : raise Exception("Server is still running. Cannot change the host and port.")
        self.__host = host
        self.__port = port
    
    # 获得连接地址
    def getHostPort(self) :
        return (self.__host, self.__port)
    
    # 接受远程信息
    def getMessage(self) :
        if not self.isConnect() : raise Exception("No server.")
        return self.__server.recv(self.__BUFSIZE).decode("utf-8")
    
    # 解析信息
    def parseMessage(self, message : str) :
        # 异常信息默认为收到攻击，停止脚本运行
        if message == "" : 
            print("Connection interrupt")
            return False
        # 传送密码
        if message[0] == AUTHENTICATION : 
            self.__sendPassword(message[2:])
            return True
        # 传送指令
        elif message[0] == INSTRUCTION :
            self.__sendInstruction()
            return True
        # 显示信息
        elif message[0] == MESSAGESHOW : 
            print(message[2:])
            return True
        # 确认行为
        elif message[0] == INSTCONFIRM :
            self.__sendInstruction(message[2:])
            return True
        elif message[0] == PASSWORDSEND :
            self.__sendMyPassword(message[2:])
            return True
        # 其他指令则默认为收到攻击，停止脚本运行
        return False
        
# 初始化
if debug : port = port + 1
server = Server(host, port)

# 尝试连接
print("host: " + host)
print("port: " + str(port))
print("Connecting to the server...")
try :
    server.connect()
except Exception :
    print("Connect failed")
    exit(0)

# 接受信息并运行
while True :
    meg = server.getMessage()
    if not server.parseMessage(meg) : break

server.close()

