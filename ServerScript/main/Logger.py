from threading import Semaphore
class Logger :
    def __init__(self, path = "./main/data/downloadlog.json"):
        self.__sem = Semaphore(1)
        self.__downloadlogpath = path
    

    def addDownloadLog(self, info : dict) :
        self.sem.acquire()
        with open(self.__downloadlogpath, "a+") as f :
            f.write(json.dumps(info))
        self.sem.release()

    def addDownloadCKP(self) :
        self.sem.acquire()
        f.write("CKP")
        self.sem.release()
    
    def CheckDownload(self) :
        last = 0
        lis = []
        with open(self.__downloadlogpath, "r") as f :
            for temp in f.readlines() :
                if temp == "CKP" : last = f.tell()
            if f.tell() == last : return
            f.seek(last)
            for temp in f.readlines() :
                lis.append(json.loads(expjson))
        return lis

            
                
        