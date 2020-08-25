#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import time, datetime
import hashlib
import sys,pefile,re,peutils,os
from hashlib import md5,sha1,sha256
import time, datetime


def help():
    print("")
    print("Usage: prog [option] file/Directory")
    print("For eg: exescan.py -a malware.exe/malware")
    print("-a","advanced scan with anomaly detection")
    print("-b","display basic information")
    print("-m","scan for commonly known malware APIs")
    print("-i","display import/export table")
    print("-p","display PE header")
    print("")

def greet():
    log("\t**********************************************************")
    log("\t**                                                      **")
    log("\t**           Cyber攻击代码库样本检索工具                **")
    log("\t**                                                      **")
    log("\t**********************************************************")

def log(data):
    #global handle
    print(data)
    #data = data
    #nextline = "\n"
    #handle.write(data)
    #handle.write(nextline)
    return

class ExeScan():
    def __init__(self,pe,file):
        self.pe = pe
        self.file = file
        self.MD5 = None
        self.SHA1 = None
        self.SHA256 = None
        self.data = None

    def hashes(self):
        f = open(self.file,"rb")
        self.data = f.read()
        self.MD5 = md5(self.data).hexdigest()
        self.SHA1 = sha1(self.data).hexdigest()
        self.SHA256 = sha256(self.data).hexdigest()
        f.close()
        return (self.MD5,self.SHA1,self.SHA256,self.data)

    def header(self):
    #header information check
        file_header = self.pe.FILE_HEADER.dump()
        log("\n")
        for i in file_header:
            log(i)
        nt_header = self.pe.NT_HEADERS.dump()
        log("\n")
        for i in nt_header:
            log(i)
        optional_header = self.pe.OPTIONAL_HEADER.dump()
        log("\n")
        for i in optional_header:
            log(i)
        log("\n")
        for i in self.pe.OPTIONAL_HEADER.DATA_DIRECTORY:
            i = i.dump()
            log("\n")
            for t in i:
                log(t)
        log("\n")
        for section in self.pe.sections:
            log("Name: %s\n" % section.Name)
            log('\tVirtual Size:            0x%.8x' % section.Misc_VirtualSize)
            log('\tVirtual Address:         0x%.8x' % section.VirtualAddress)
            log('\tSize of Raw Data:        0x%.8x' % section.SizeOfRawData)
            log('\tPointer To Raw Data:     0x%.8x' % section.PointerToRawData)
            log('\tPointer To Relocations:  0x%.8x' % section.PointerToRelocations)
            log('\tPointer To Linenumbers:  0x%.8x' % section.PointerToLinenumbers)
            log('\tNumber Of Relocations:   0x%.8x' % section.NumberOfRelocations)
            log('\tNumber Of Linenumbers:   0x%.8x' % section.NumberOfLinenumbers)
            log('\tCharacteristics:         0x%.8x\n' % section.Characteristics)

    def anomalis(self):
        log("\n[+] 异常信息检测\n")
        a_labels = []
        a_contents = []
	# Entropy based check.. imported from peutils
        pack = peutils.is_probably_packed(self.pe)
        if pack == 1:
            log("\t[*] Based on the sections entropy check! file is possibly packed")
            a_labels.append("打包检测")
            a_contents.append("Based on the sections entropy check! file is possibly packed")
	# SizeOfRawData Check.. some times size of raw data value is used to crash some debugging tools.
        nsec = self.pe.FILE_HEADER.NumberOfSections
        for i in range(0, nsec-1):
            if i == nsec-1:
                break
            else:
                nextp = self.pe.sections[i].SizeOfRawData + self.pe.sections[i].PointerToRawData
                currp = self.pe.sections[i+1].PointerToRawData
                if nextp != currp:
                    log("\t[*] The Size Of Raw data is valued illegal! Binary might crash your disassembler/debugger")
                    a_labels.append("原始数据大小检测")
                    a_contents.append("The Size Of Raw data is valued illegal! Binary might crash your disassembler/debugger")
                    break
                else:
                    pass

	# Non-Ascii or empty section name check
        for sec in self.pe.sections:
            print(5555)
            sec.Name = str(sec.Name)
            print(sec.Name)
            if not re.match("^[.A-Za-z][a-zA-Z]+",sec.Name):
                log("\t[*] Non-ascii or empty section names detected")
                a_labels.append("非Ascii或空节名检测")
                a_contents.append("Non-ascii or empty section names detected")
                break;

	# Size of optional header check
        if self.pe.FILE_HEADER.SizeOfOptionalHeader != 224:
            log("\t[*] Illegal size of optional Header")
            a_labels.append("可选头大小检测")
            a_contents.append("Illegal size of optional Header")


	# Zero checksum check
        if self.pe.OPTIONAL_HEADER.CheckSum == 0:
            log("\t[*] Header Checksum is zero!")
            a_labels.append("零校验")
            a_contents.append("Header Checksum is zero!")

	# Entry point check
        enaddr = self.pe.OPTIONAL_HEADER.AddressOfEntryPoint
        vbsecaddr = self.pe.sections[0].VirtualAddress
        ensecaddr = self.pe.sections[0].Misc_VirtualSize
        entaddr = vbsecaddr + ensecaddr
        if enaddr > entaddr:
            log("\t[*] Enrty point is outside the 1st(.code) section! Binary is possibly packed")
            a_labels.append("入口点检测")
            a_contents.append("Enrty point is outside the 1st(.code) section! Binary is possibly packed")
	# Numeber of directories check
        if self.pe.OPTIONAL_HEADER.NumberOfRvaAndSizes != 16:
            log("\t[*] Optional Header NumberOfRvaAndSizes field is valued illegal")
            a_labels.append("目录号检测")
            a_contents.append("Optional Header NumberOfRvaAndSizes field is valued illegal")
	# Loader flags check
        if self.pe.OPTIONAL_HEADER.LoaderFlags != 0:
            log("\t[*] Optional Header LoaderFlags field is valued illegal")
            a_labels.append("加载标志检测")
            a_contents.append("Optional Header LoaderFlags field is valued illegal")
	# TLS (Thread Local Storage) callback function check
        if hasattr(self.pe,"DIRECTORY_ENTRY_TLS"):
            log("\t[*] TLS callback functions array detected at 0x%x" % self.pe.DIRECTORY_ENTRY_TLS.struct.AddressOfCallBacks)
            a_labels.append("TLS回调函数检查")
            a_contents.append("TLS callback functions array detected at"+self.pe.DIRECTORY_ENTRY_TLS.struct.AddressOfCallBacks)
            callback_rva = self.pe.DIRECTORY_ENTRY_TLS.struct.AddressOfCallBacks - self.pe.OPTIONAL_HEADER.ImageBase
            log("\t[*] Callback Array RVA 0x%x" % callback_rva)
            a_labels.append("TLS回调函数检查")
            a_contents.append("Callback Array RVA"+callback_rva)

        a_title = "异常信息检测"
        return a_title,a_labels,a_contents
        
    def base(self,check):
        log("\n[+] 编译器和加壳信息检测\n")
        if check:
            for i in check:
                log('\t%s' % i)
        else:
            check = "No match found"
            log("\t[*] No match found.\n")
        
        log("\n[+] 程序入口点   : 0x%.8x\n" % self.pe.OPTIONAL_HEADER.AddressOfEntryPoint)
        log("[+] 镜像基址       : 0x%.8x\n" % self.pe.OPTIONAL_HEADER.ImageBase)
        log("[+] 节表")
        s_contents = []
        for section in self.pe.sections:
            log("\t节名: %s\t" % section.Name.strip() + "虚拟地址: 0x%.8x\t" % section.VirtualAddress + "大小: 0x%.8x\t" % section.Misc_VirtualSize + "节的信息熵: %f" % section.get_entropy())
            s_content = [section.Name.strip(),section.VirtualAddress,section.Misc_VirtualSize,section.get_entropy()]
            s_contents.append(s_content)
        s_title = "节表"
        f_labels = ["编译器和加壳信息检测","程序入口点","镜像基址"]
        f_contents = [check,self.pe.OPTIONAL_HEADER.AddressOfEntryPoint,self.pe.OPTIONAL_HEADER.ImageBase]
        s_labels = ["节名","虚拟地址","大小","节的信息熵"]
        return f_labels, f_contents, s_labels, s_contents,s_title
    def importtab(self):
        if hasattr(self.pe,"DIRECTORY_ENTRY_IMPORT"):
            log("\n[+] 引入表\n")
            for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
                log('\n[-] %s\n' % entry.dll)
                for imp in entry.imports:
                    log('\t0x%.8x\t%s' % (imp.address, imp.name))

    def exporttab(self):
        if hasattr(self.pe,"DIRECTORY_ENTRY_EXPORT"):
            log("\n[+] 引出表\n")
            for entry in self.pe.DIRECTORY_ENTRY_EXPORT.symbols:
                log('\t0x%.8x\t%s' % (entry.address, entry.name))

    def malapi(self,MALAPI,str1):
        #print(str1)
        dict1 = {}
        i_labels=[]
        i_contents=[]
        i_contents_=[]
        e_labels=[]
        e_contents=[]
        log("\n[+] 可疑API检测\n")
        log("\n\t[-] 引入表搜索\n")
        i_title =""
        for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if imp.name != None:
                    dict1[imp.name] = imp.address
        dict1 = {k.decode('utf-8'): v for k, v in dict1.items()}
        #print(dict1.keys())
        i=0
        for m in MALAPI:
            m = m.strip()
            #print(m)
            print(343434)
            if m in dict1.keys():
                print(454545)
                i_title ="引入表搜索"
                newStr= "IA_{0}".format(i)
                i_labels.append(newStr)
                i = i+1
                #print(type(dict1[m]))
                i_contents.append(dict1[m])
                i_contents_.append(m)
                print(8888888)
                print(m)
                #print(i_contents_)
                log("\t\tIA: 0x%08x\t%s" % (dict1[m],m))
        log("\n\t[-] 整个可执行程序搜索\n")
        e_title =""
        d=""
        j=0
        for m_ in MALAPI:
            i = 0
            m_ = m_.strip()
            #print(len(str1))
            for s in str1:
                #s= s.decode('utf-8')
                #s=bytes.decode(s)
                #print(type(s))
                #print(s)
                #print(type(s))
                #s2 = str(s, encoding="utf-8")
                #s2 = bytes.decode(s)
                #print(s2)
                #print(type(s2))
                #print(m_)
                #print(type(m))
                if re.search(m_,s):
                   print(101010)
                   d = m_
                   i = i + 1
            print(898989)
            if d == m_:
                e_title ="整个可执行程序搜索"
                j=j+1
                e_labels.append('{0} times_{1}'.format(i,j))
                e_contents.append(d)
                log("\t\t %d times\t%s" % (i,d))
        print(444444444)
        print(e_labels)
        print(e_contents)
        return i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents

class StringAndThreat():
    def __init__(self,MD5,data):
        self.MD5 = MD5
        self.data = data
        #self.handle = None

    def StringE(self):
        name = "strings_"+self.MD5+".txt"
        name = os.path.join(self.MD5,name)
        if os.path.exists(name):
            return
        #self.handle = open(name,'a')
        headline = "\t\t\tStrings-%s\n\n" % self.MD5
        #self.handle.write(headline)
        #for m in re.finditer("([\x20-\x7e]{3,})", self.data):
            #self.handle.write(m.group(1))
            #self.handle.write("\n")
        return

def main_s(pe,f,name):
    #global handle
    filesize = os.stat(name).st_size
    filesize = filesize/1024
    filesize = str(filesize)+"KB"

    atime = os.stat(name).st_atime
    mtime = os.stat(name).st_mtime
    ctime = os.stat(name).st_ctime
    dateArray = datetime.datetime.fromtimestamp(atime)
    atime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
    dateArray = datetime.datetime.fromtimestamp(mtime)
    mtime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
    dateArray = datetime.datetime.fromtimestamp(ctime)
    ctime = dateArray.strftime("%Y--%m--%d %H:%M:%S")

    exescan = ExeScan(pe,name)
    (MD5,SHA1,SHA256,data) = exescan.hashes()
    stx = StringAndThreat(MD5,data)
    # store reports in folders
    #if os.path.exists(MD5):
    #    report_name = str(MD5)+".txt"
    #    report_name = os.path.join(MD5,report_name)
    #else:
    #    os.mkdir(MD5)
    #    report_name = str(MD5)+".txt"
    #    report_name = os.path.join(MD5,report_name)
    #handle = open(report_name,'a')
    greet()
    filetype=None
    f_contents = []
    print(type(name))
    '''
    log("\n\n[+] 文件位置 : %s" %name)
    log("\n\t[*] MD5    : %s" %MD5)
    log("\t[*] SHA-1    : %s" %SHA1)
    log("\t[*] SHA-256  : %s" %SHA256)
    log("\t[*] 文件访问时间      : %s" % atime)
    log("\t[*] 文件内容修改时间      : %s" % mtime)
    #log("\t[*] 文件属性修改时间      : %s" % ctime)
    log("\t[*] 文件大小      : %s" % filesize)
    '''
    f_labels = ["文件位置","MD5","SHA-1","SHA-256","文件访问时间","文件内容修改时间","文件大小","文件类型"]
    #check file type (exe, dll)
    if pe.is_exe():
        filetype = "EXE"
        log("\n[+] 文件类型: EXE")
    elif pe.is_dll():
        filetype = "DLL"
        log("\n[+] 文件类型: DLL")
    else:
        filetype=[]
        log("\n 不是PE文件结构")
        return 2
    f_contents.append(name)
    f_contents.append(MD5)
    f_contents.append(SHA1)
    f_contents.append(SHA256)
    f_contents.append(atime)
    f_contents.append(mtime)
    f_contents.append(filesize)
    f_contents.append(filetype)
    strings = f.readlines()
    mf = open("API.txt","r")
    MALAPI = mf.readlines()
    signature  = peutils.SignatureDatabase("userdb.txt")
    check = signature.match_all(pe,ep_only = True)

    flabels,fcontents,s_labels,s_contents,s_title=exescan.base(check)
    #exescan.header()
    #exescan.importtab()
    #exescan.exporttab()
    f_labels.extend(flabels)
    f_contents.extend(fcontents)
    i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents=exescan.malapi(MALAPI,strings)
    a_title,a_labels,a_contents=exescan.anomalis()
    #stx.StringE()

    #if ch == "-i":
    #    exescan.base(check)
    #    exescan.importtab()
    #    exescan.exporttab()
    #elif ch == "-b":
    #    exescan.base(check)
    #elif ch == "-m":
    #    exescan.base(check)
    #    exescan.malapi(MALAPI,strings)
    #elif ch == "-p":
    #    exescan.base(check)
    #    exescan.header()
    #elif ch == "-a":
    #    exescan.base(check)
    #    exescan.anomalis()
    #    exescan.malapi(MALAPI,strings)
    #    stx.StringE()
    #else:
    #    print()
    mf.close()
    #handle.close()
    return f_labels,f_contents,s_labels,s_contents,s_title,a_title,a_labels,a_contents,i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents

def pe_info(fname):
    folder = "../DATA/" + fname[0] + "/"+ fname[1] + "/"+ fname[2]+ "/" + fname[3] + "/"
    fname = folder + fname
    str_cmd = "file {}".format(fname)
    filetype =  os.popen(str_cmd).read().strip().split("/")[-1]
    fname = os.path.realpath(fname)
    #print(fname)
    pe = pefile.PE(fname)
    f = open(fname,"rb")
    labels,contents,s_labels,s_contents,s_title,a_title,a_labels,a_contents,i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents = main_s(pe, f, fname)
    f.close()
    pe.__data__.close()
    return labels,contents,s_labels,s_contents,s_title,a_title,a_labels,a_contents,i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents
 #    #except Exception, WHY:
    #print("\nInvalid PE file\n")
    #print("Verbose: %s" % WHY)

def file_info(folder,s,path,filetype, algorithm):
    global start, end  # 声明全局变量
    start = time.time()  # 获取当前时间，用于记录计算过程的耗时
    size = os.path.getsize(path)  # 获取文件大小，单位是字节（byte）
    with open(path, 'rb') as f:  # 以二进制模式读取文件
        while size >= 1024 * 1024:  # 当文件大于1MB时将文件分块读取
            algorithm.update(f.read(1024 * 1024))
            size -= 1024 * 1024
        algorithm.update(f.read())
    end = time.time()  # 获取计算结束后的时间
    md5 = algorithm.hexdigest()
    sha1 = algorithm.hexdigest()
    contents = []    
    filesize = os.stat(path).st_size
    filesize = filesize/1024
    filesize = str(filesize)+"KB"
    accesstime = os.stat(path).st_atime
    modifytime = os.stat(path).st_mtime
    changetime = os.stat(path).st_ctime
    dateArray = datetime.datetime.fromtimestamp(accesstime)
    accesstime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
    dateArray = datetime.datetime.fromtimestamp(modifytime)
    modifytime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
    dateArray = datetime.datetime.fromtimestamp(changetime)
    changetime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
    filelocation = os.path.abspath(folder)
    fileinfo = "SHA256?"+s+"?"+"MD5?"+md5+"?"+"SHA1?"+sha1+"?"+"文件位置?"+filelocation+"?"+"文件类型?"+filetype+"?"+"文件大小?"+str(filesize)+"?"+"文件访问时间?"+str(accesstime)+"?"\
    +"文件内容修改时间?"+str(modifytime)
    labels = ["SHA-256","MD-5","SHA-1","文件位置","文件类型","文件大小","文件访问时间"+"文件内容修改时间"]
    contents.append(s)
    contents.append(md5)
    contents.append(sha1)
    contents.append(filelocation)
    contents.append(filetype)
    contents.append(filesize)
    contents.append(accesstime)
    contents.append(modifytime)
    s_labels =[]
    s_contents = []
    s_title=""
    a_title=""
    a_labels=[]
    a_contents=[]
    i_title=""
    i_labels=[]
    i_contents=[]
    i_contents_=[]
    e_title=""
    e_labels=[]
    e_contents=[]
    return labels,contents,s_labels,s_contents,s_title,a_title,a_labels,a_contents,i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents
    
    
def get_sha256_info(s):
    folder = "../DATA/" + s[0] + "/"+ s[1] + "/"+ s[2]+ "/" + s[3] + "/"
    f_path = folder + s
    if not os.path.exists(os.path.abspath(f_path)):
        fileinfo = s+"?"+"Cyber攻击代码库没有此文件"
        print(fileinfo)
        title = "文件基本信息"
        f_labels = [s]
        f_contents = ["Cyber攻击代码库没有此文件"]
        s_labels = []
        s_contents = []
        s_title = ""
        a_title = ""
        a_labels = []
        a_contents = []
        i_title=""
        i_labels=[]
        i_contents=[]
        i_contents_=[]
        e_title=""
        e_labels=[]
        e_contents=[]
    else:
        str_cmd = "file {}".format(f_path) 
        filetype =  os.popen(str_cmd).read().strip().split("/")[-1]
        filetype = filetype.split(":")[-1]
        print(filetype)
        print(type(filetype))
        print("GUI" is filetype)
        print(filetype.find("PE32"))
        if filetype.find("PE32") != -1:
            print(1111)
            title = "PE文件基本信息"
            f_labels,f_contents,s_labels,s_contents,s_title,a_title,a_labels,a_contents,i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents = pe_info(s)
        else:
            print(222)
            title = "文件基本信息"
            f_labels,f_contents,s_labels,s_contents,s_title,a_title,a_labels,a_contents,i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents=file_info(folder,s,f_path,filetype,hashlib.md5())
    return title,f_labels,f_contents,s_labels,s_contents,s_title,a_title,a_labels,a_contents,i_title,i_labels,i_contents,i_contents_,e_title,e_labels,e_contents
   
def parseargs():
    parser = argparse.ArgumentParser(description = "to get samples info from sha256")
    parser.add_argument("-s", "--sha256", help="input sha256", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseargs()
    get_sha256_info(args.sha256)

