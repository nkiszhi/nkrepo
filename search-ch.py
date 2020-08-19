#!/usr/bin/env python
#-*- coding: utf-8 -*-
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
    log("\t\t**********************************************************")
    log("\t\t**                                                      **")
    log("\t\t**           Cyber攻击代码库样本检索工具                **")
    log("\t\t**                                                      **")
    log("\t\t**********************************************************")


def log(data):
    global handle
    print(data)
    data = data
    nextline = "\n"
    handle.write(data)
    handle.write(nextline)
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
 
        # Entropy based check.. imported from peutils
        pack = peutils.is_probably_packed(self.pe)
        if pack == 1:
            log("\t[*] Based on the sections entropy check! file is possibly packed")
 
        # SizeOfRawData Check.. some times size of raw data value is used to crash some debugging tools.
        nsec = self.pe.FILE_HEADER.NumberOfSections
        for i in range(0,nsec-1):
            if i == nsec-1:
                break
            else:
                nextp = self.pe.sections[i].SizeOfRawData + self.pe.sections[i].PointerToRawData
                currp = self.pe.sections[i+1].PointerToRawData
                if nextp != currp:
                    log("\t[*] The Size Of Raw data is valued illegal! Binary might crash your disassembler/debugger")
                    break
                else:
                    pass

    # Non-Ascii or empty section name check
        for sec in self.pe.sections:
            if not re.match("^[.A-Za-z][a-zA-Z]+",sec.Name):
                log("\t[*] Non-ascii or empty section names detected")
                break;

        # Size of optional header check
        if self.pe.FILE_HEADER.SizeOfOptionalHeader != 224:
            log("\t[*] Illegal size of optional Header")
        
        # Zero checksum check
        if self.pe.OPTIONAL_HEADER.CheckSum == 0:
            log("\t[*] Header Checksum is zero!")
        
        # Entry point check
        enaddr = self.pe.OPTIONAL_HEADER.AddressOfEntryPoint
        vbsecaddr = self.pe.sections[0].VirtualAddress
        ensecaddr = self.pe.sections[0].Misc_VirtualSize
        entaddr = vbsecaddr + ensecaddr
        if enaddr > entaddr:
            log("\t[*] Enrty point is outside the 1st(.code) section! Binary is possibly packed")
        
        # Numeber of directories check
        if self.pe.OPTIONAL_HEADER.NumberOfRvaAndSizes != 16:
            log("\t[*] Optional Header NumberOfRvaAndSizes field is valued illegal")
        
        # Loader flags check
        if self.pe.OPTIONAL_HEADER.LoaderFlags != 0:
            log("\t[*] Optional Header LoaderFlags field is valued illegal")

        # TLS (Thread Local Storage) callback function check
        if hasattr(self.pe,"DIRECTORY_ENTRY_TLS"):
            log("\t[*] TLS callback functions array detected at 0x%x" % self.pe.DIRECTORY_ENTRY_TLS.struct.AddressOfCallBacks)
            callback_rva = self.pe.DIRECTORY_ENTRY_TLS.struct.AddressOfCallBacks - self.pe.OPTIONAL_HEADER.ImageBase
            log("\t[*] Callback Array RVA 0x%x" % callback_rva)
         
    def base(self,check):
        log("\n[+] 编译器和加壳信息检测\n")
        if check:
            for i in check:
                log('\t%s' % i)
        else:
            log("\t[*] No match found.\n")

        log("\n[+] 程序入口点	: 0x%.8x\n" % self.pe.OPTIONAL_HEADER.AddressOfEntryPoint)
        log("[+] 镜像基址	: 0x%.8x\n" % self.pe.OPTIONAL_HEADER.ImageBase)
        log("[+] 节表")
        for section in self.pe.sections:
            log("\t节名: %s\t" % section.Name.strip() + "虚拟地址: 0x%.8x\t" % section.VirtualAddress + "大小: 0x%.8x\t" % section.Misc_VirtualSize + "节的信息熵: %f" % section.get_entropy())

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
 
    def malapi(self,MALAPI,str):
        dict = {}
        log("\n[+] 可疑API检测\n")
        log("\n\t[-] 引入表搜索\n")
        for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                dict[imp.name] = imp.address
        for m in MALAPI:
            m = m.strip()
            if m in dict.keys():
                log("\t\tIA: 0x%08x\t%s" % (dict[m],m))
        log("\n\t[-] 整个可执行程序搜索\n")
        for m in MALAPI:
            i = 0
            m = m.strip()
            try:
                for s in str:
                    if re.search(m,s):
                        d = m
                        i = i + 1
                if d == m:
                    log("\t\t %d times\t%s" % (i,d))
            except:
                pass



class StringAndThreat():
    def __init__(self,MD5,data):
        self.MD5 = MD5
        self.data = data
        self.handle = None
  
    def StringE(self):
        name = "strings_"+self.MD5+".txt"
        name = os.path.join(self.MD5,name)
        if os.path.exists(name):
            return
        self.handle = open(name,'a')
        headline = "\t\t\tStrings-%s\n\n" % self.MD5
        self.handle.write(headline)
        for m in re.finditer("([\x20-\x7e]{3,})", self.data):
            self.handle.write(m.group(1))
            self.handle.write("\n")
        return



def main_s(pe,ch,f,name):
    global handle
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
    if os.path.exists(MD5):
        report_name = str(MD5)+".txt"
        report_name = os.path.join(MD5,report_name)
    else:
        os.mkdir(MD5)
        report_name = str(MD5)+".txt"
        report_name = os.path.join(MD5,report_name)
    handle = open(report_name,'a')
    greet()
    log("\n\n[+] 文件名 : %s" % name)
    log("\n\t[*] MD5 	: %s" % MD5)
    log("\t[*] SHA-1 	: %s" % SHA1)
    log("\t[*] SHA-256	: %s" % SHA256)
    log("\t[*] 文件访问时间      : %s" % atime)
    log("\t[*] 文件内容修改时间      : %s" % mtime)
    log("\t[*] 文件属性修改时间      : %s" % ctime)
    log("\t[*] 文件大小      : %s" % filesize)
    #check file type (exe, dll)
    if pe.is_exe():
        log("\n[+] 文件类型: EXE")
    elif pe.is_dll():
        log("\n[+] 文件类型: DLL")
    else:
        log("\n 不是PE文件结构")
    strings = f.readlines()
    mf = open("API.txt","r")
    MALAPI = mf.readlines()
    signature  = peutils.SignatureDatabase("userdb.txt")
    check = signature.match_all(pe,ep_only = True)

    exescan.base(check)
    #exescan.header()
    #exescan.importtab()
    #exescan.exporttab()
    exescan.malapi(MALAPI,strings)
    exescan.anomalis()
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
    handle.close()
    return MD5

def main():
    if len(sys.argv) < 3:
        help()
        sys.exit(0)
    ch = sys.argv[1]
    fname = sys.argv[2]
    folder = "DATA/" + fname[0] + "/"+ fname[1] + "/"+ fname[2]+ "/" + fname[3] + "/"
    fname = folder + fname
    str_cmd = "file {}".format(fname)
    filetype =  os.popen(str_cmd).read().strip().split("/")[-1]
    #print(filetype)
    #if os.path.isdir(fname):
    #    filelist = os.listdir(fname)
    #    for name in filelist:
    #        try:
    #            name = os.path.join(fname,name)
    #            pe = pefile.PE(name)
    #            f = open(name,"rb")
    #            new_name = main_s(pe,ch,f,name)
    #            f.close()
    #            pe.__data__.close()
    #            try:
    #                new_name = new_name + ".bin"
    #                new_name = os.path.join(fname,new_name)
    #                os.rename(name,new_name)
    #            except:
    #                pass
    #        except:
    #            pass
    #else:
    #    try:
    #        fname = os.path.realpath(fname)
    #        print(fname)
    #        pe = pefile.PE(fname)
    #        f = open(fname,"rb")
    #        new_name = main_s(pe,ch,f,fname)
    #        f.close()
    #        pe.__data__.close()
    #    #except Exception, WHY:
    #    except:
    #        print("\nInvalid file\n")
    #        #print("Verbose: %s" % WHY)
    #        sys.exit(0)


    try:
        fname = os.path.realpath(fname)
        #print(fname)
        pe = pefile.PE(fname)
        f = open(fname,"rb")
        new_name = main_s(pe, ch, f, fname)
        f.close()
        pe.__data__.close()
     #    #except Exception, WHY:
    except:
        #print("\nInvalid PE file\n")
        #print("Verbose: %s" % WHY)
        sys.exit(0)

if __name__ == '__main__':
    main()
