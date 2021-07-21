import paramiko
import scp
import os
import time

# 远程目录
remote_path="/home/nkamg/VirusDatabase/AutoScript/"

#服务器ip地址
host = "10.134.78.10"
# host = "10.134.75.233"
# 端口
port = 22

# 用户名
username = "nkamg"

# 密码
password = "nkamg" 

# 需要传输的文件或者文件夹
# local_paths = ["backstage", "webstart.py", "templates", "static", "webdata"]
local_paths = ["nkaz", "nkvs", "nkvt", "main", "scriptmain.py", "backstage", "webstart.py", "templates", "static", "webdata"]

# 连接
print("Connecting...")
sshclient = paramiko.SSHClient()
sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy)
sshclient.connect(host, username = username, password = password, port = port)
scpclient = scp.SCPClient(sshclient.get_transport(),socket_timeout=15.0)

# 传输文件
print("Connected!")
print("Start to update")
for files in local_paths :
    if os.path.exists(files) :
        print("Updating: " + files)
        scpclient.put(files, remote_path = remote_path + files + "_temp", recursive = os.path.isdir(files))
    else :
        print("Cannot find the file or directory: " + files)
        print("Update interrupt")
        for fil in local_paths :
            if fil == files : break
            sshclient.exec_command("rm" + (" -r " if os.path.isdir(fil) else ' ') + remote_path + fil + "_temp")
        exit()

for files in local_paths :
    sshclient.exec_command("rm" + (" -r " if os.path.isdir(files) else ' ') + remote_path + files)
    sshclient.exec_command("mv " + remote_path + files + "_temp " + remote_path + files)
print("Update successfully!")

# 重启网站
# print("Restarting the website...")
# sshclient.exec_command("cd " + remote_path + ';' + "nohup python3 webstart.py > nohup_log/nohup_web.log 2>&1 &")
# print("Complete!")

# 断开连接
sshclient.close()
print("Exit")