#!/usr/bin/env python
#-*- coding: utf-8 -*-
import sys,pefile,re,peutils,os
from hashlib import md5,sha1,sha256

def log(data):
		#global handle
		print (data)
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
                #print(str)
		dict2 = {}
		log("\n[+] 恶意API探测\n")
		log("\n\t[-] 输入表\n")
		for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
			for imp in entry.imports:
				dict2[imp.name] = imp.address
                #print(dict.keys())
		for m in MALAPI:
			m = m.strip()
			if m in dict2.keys():
				#print(1111)
				#print(m)
				log("\t\tIA: 0x%08x\t%s" % (dict2[m],m))
		log("\n\t[-] 整个表\n")
                d=""
		for m_ in MALAPI:
			i = 0
			m_ = m_.strip()
                        print(len(str2))
			for s in str2:
				#print(s)
                                #print(787878)
				#print(type(s))
				#print(m_)
				if re.search(m_,s):
					#print(898989)
					d = m_
                                        #print(s)
                                        #print(type(s))
                                        #print(m)
					i = i + 1
			if d == m_:
				#print(101010)
				log("\t\t %d times\t%s" % (i,d))
                                #print(m)
				#print(type(s))
                                #print(s)
				

				

def main_s(pe,ch,f,name):
	#global handle
	exescan = ExeScan(pe,name)
	# store reports in folders
	'''
	if os.path.exists(MD5):
		report_name = str(MD5)+".txt"
		report_name = os.path.join(MD5,report_name)
	else:
		os.mkdir(MD5)
		report_name = str(MD5)+".txt"
		report_name = os.path.join(MD5,report_name)
	handle = open(report_name,'a')
        '''
	strings = f.readlines()
	mf = open("API.txt","r")
	MALAPI = mf.readlines()
	signature  = peutils.SignatureDatabase("userdb.txt")
	check = signature.match_all(pe,ep_only = True)
	'''
	if ch == "-i":
		exescan.base(check)
		exescan.importtab()
		exescan.exporttab()
	elif ch == "-b":
		exescan.base(check)
	elif ch == "-m":
		exescan.base(check)
		exescan.malapi(MALAPI,strings)
	elif ch == "-p":
		exescan.base(check)
		exescan.header()
        '''
	exescan.malapi(MALAPI,strings)
	mf.close()
def main():
	if len(sys.argv) < 2:
		help()
		sys.exit(0)
	ch = sys.argv[1]
	fname = sys.argv[1]
	fname = os.path.realpath(fname)
	print (fname)
	pe = pefile.PE(fname)
	f = open(fname,"rb")
	new_name = main_s(pe,ch,f,fname)
	f.close()
	pe.__data__.close()

if __name__ == '__main__':
		main()
