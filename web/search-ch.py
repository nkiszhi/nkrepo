#!/usr/bin/env python
#-*- coding: utf-8 -*-
import sys,pefile,re,peutils,os
from hashlib import md5,sha1,sha256
import time, datetime

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

        
 
    def malapi(self,MALAPI,str2):
        dict2 = {}
        print(999999)
        log("\n[+] 可疑API检测\n")
        log("\n\t[-] 引入表搜索\n")
        for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
           # print(entry)
            #print(8888888)
            for imp in entry.imports:
                dict2[imp.name] = imp.address
        #print(MALAPI)
        for m in MALAPI:
           # print(0000000)
            m = m.strip()
            #print(7777)
            #print(m)
            if m in dict2.keys():
                log("\t\tIA: 0x%08x\t%s" % (dict2[m],m))
        log("\n\t[-] ???\n")
        d=""
        for m_ in MALAPI:
            i = 0
            m_ = m_.strip()
            #print(type(m))
            #print(len(str2))
            for s in str2:
                print(s)
                #log("\t\t %s\t" %type(s))
                print(type(s))
                #str3 = str(b, encoding = "utf-8") 
                #str3 = str.encode(s)
                #print(str3)
                #print(type(m))
                
                if re.search(m_,s):
                    #print(type(s))
                    #print(type(m_))
                    #print(11111)
                    d = m_
                    i = i + 1
            if d == m_:
                print(s)
                print(type(s))
                #byte1 = s.encode("utf-8")
                #byte1=bytes(s, encoding = "unicode")
                #print(byte1)
                log("\t\t %d times\t%s" % (i,d))
                



def main_s(pe,ch,f,name):
    
    exescan = ExeScan(pe,name)
    # store reports in folders
    #if os.path.exists(MD5):
    #    report_name = str(MD5)+".txt"
    #    report_name = os.path.join(MD5,report_name)
    #else:
    #    os.mkdir(MD5)
    #    report_name = str(MD5)+".txt"
    #    report_name = os.path.join(MD5,report_name)
    #handle = open(report_name,'a')
    strings = f.readlines()
    #print(strings)
    #sys.exit(0)
    mf = open("API.txt","r")
    MALAPI = mf.readlines()
    signature  = peutils.SignatureDatabase("userdb.txt")
    check = signature.match_all(pe,ep_only = True)

    #exescan.header()
    #exescan.importtab()
    #exescan.exporttab()
    #exescan.anomalis()
    exescan.malapi(MALAPI,strings)
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
    return MD5

def main():
    if len(sys.argv) < 2:
        help()
        sys.exit(0)
    ch = sys.argv[1]
    print(ch)
    fname = sys.argv[1]
    folder = "../DATA/" + fname[0] + "/"+ fname[1] + "/"+ fname[2]+ "/" + fname[3] + "/"
    fname = folder + fname
    #str_cmd = "file {}".format(fname)
    #filetype =  os.popen(str_cmd).read().strip().split("/")[-1]
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
