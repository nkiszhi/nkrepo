#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib, os, shutil
from multiprocessing import Pool
import paramiko  # 用于调用scp命令
import scp
 
hexstring="0123456789abcdef"
secondstring="1"      ###########
# 将服务器指定目录下载到本机指定目录
# remote_path远程服务器文件夹
# file_path本地文件夹路径
def download_dir(remote_path, local_path="D:\\scp_get\\"):##########
    host = "10.134.75.90"  #服务器ip地址
    port = 1001  # 端口号
    username = "nkamg"  # ssh 用户名
    password = "1012"  # 密码
 
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(host, port, username, password,allow_agent=False,look_for_keys=False)
    scpclient = scp.SCPClient(ssh_client.get_transport(),socket_timeout=300.0)
    
    try:
        scpclient.get(remote_path, local_path,recursive=True)
    except FileNotFoundError as e:
        print(e)
        print("系统找不到指定文件" + remote_path)
    else:
        print("文件上传成功")
    ssh_client.close()
    
def main():
    rootpath="/nkrepo/DATA/sha256"
    # 添加所有路径
    folderList=[]
    for j in secondstring:
        for k in hexstring:
            for l in hexstring:
                cpath=rootpath+"/0/"+j+"/"+k+"/"+l###########
                folderList.append(cpath)
    print(folderList[0])
    print("folder finish")

    p=Pool(10)
    p.map(download_dir,folderList)
    print("finish!!!!")


if __name__ == "__main__":
    main()
