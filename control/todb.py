import os
import sqlite3
import time
import shutil
import json
import hashlib

HEX_STRING = "0123456789abcdef"
ROOTPATH = './sha256/'

def get_hash(file_path: str, hash_method) -> str:
    if not os.path.exists(file_path):
        print("Not existed: " + file_path)
        return ''
    h = hash_method()
    with open(file_path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b: break
            h.update(b)
    return h.hexdigest()

def get_sha1(file_path: str) -> str:
    return get_hash(file_path, hashlib.sha1)

def get_md5(file_path: str) -> str:
    return get_hash(file_path, hashlib.md5)

def time_localtime(timestamp):
    stime = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d %H:%M:%S', stime)

def get_createtime(fpath):
    fpath = ROOTPATH + fpath[0] + "/" + fpath[1] + "/" + fpath[2] + "/" + fpath[3] + "/" + fpath[4] + "/" + fpath
    t = os.path.getctime(fpath)
    return time_localtime(t)

def get_info_by_sha256(sha256):
    # 1. Get json file location
    f_json = ROOTPATH + sha256[0] + "/"+ sha256[1] + "/"+ sha256[2]+ "/" + sha256[3] + "/" + sha256[4] + "/" + sha256 + ".json"
    # 2. Check if json file is existed
    if not os.path.exists(f_json):
        return {} # If json file is not existed return -1
    # 3. Read json file
    with open(f_json, "r") as f:
        str_f = f.read()
        if len(str_f) > 0:
            dict_json = json.loads(str_f)
        else:
            dict_json = {}
    # 4. return json info
    return dict_json

def search(sha256):
    print(sha256)
    dict_json = get_info_by_sha256(sha256)
    fpath = ROOTPATH + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256[4] + "/" + sha256
    sha1 = get_sha1(fpath)
    md5 = get_md5(fpath)
    if len(dict_json.keys()) == 2:
        scans = dict_json["results"]
        if "scans" not in scans:
            scans={"Kaspersky": {"detected": False,"result": "nodata.nodata.nodata"}}
        else:
            scans = scans["scans"]
    elif "scans" in dict_json:
        scans = dict_json["scans"]
    else:
        scans={"Kaspersky": {"detected": False,"result": "nodata.nodata.nodata"}}

    if "Kaspersky" not in scans:
        kav_result = "nodata.nodata.nodata"
        updatetime = "00000000"
    else:
        if scans["Kaspersky"]['detected']:
            kav_result = scans["Kaspersky"]['result']
            updatetime = scans["Kaspersky"]['update']
        else:
            kav_result = "CLEAN.nodata.nodata"
            updatetime = "00000000"
    # AdWare.MSIL.Ocna.aps
    list_kav = kav_result.split(".")
    if len(list_kav) >=3 :
        category = list_kav[0]
        platform = list_kav[1]
        family = list_kav[2]
        name = kav_result
    else:
        print("!")
        category = kav_result
        platform = "nodata"
        family = "nodata"
        
    filesize = os.path.getsize(fpath)
    timee = get_createtime(sha256)
    result = [sha256,md5,sha1,timee,updatetime,family,platform,category,filesize,name]
    return result


def main1():
    conn = sqlite3.connect('data4.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE test1
           ( SHA256  CHAR(64) NOT NULL,
             MD5     CHAR(32),
             SHA1    CHAR(40),
             TIMEE   TEXT    NOT NULL,
             FTIME   TEXT,
             FLAG    INT,
             FAMILY  CHAR(200),
             PLATT   INT,
             PLATF   CHAR(32),
             CATEG   CHAR(32),
             LEN     INT,
             VIRNAME CHAR(1000),            
             STC     CHAR(50),
             VTC     CHAR(50),
             VTN     CHAR(200),
             ADR     CHAR(200),
             FILETYPE CHAR(200)
             );''')
    conn.commit()
    n = 0 
    for i in HEX_STRING:
        for j in HEX_STRING:
            for k in HEX_STRING:
                for l in HEX_STRING:
                    for m in HEX_STRING:
                        file_list = os.listdir(ROOTPATH + i + "/" + j + "/" + k + "/" + l + "/" + m + "/")
                        file_list = list(filter(lambda x: len(x) == 64, file_list))
                        other_list = map(search, file_list)
                        firstzip = list(zip(*other_list))
                        secondzip = zip(firstzip[0], firstzip[1], firstzip[2], firstzip[3], firstzip[4], firstzip[5], firstzip[6], firstzip[7], firstzip[8], firstzip[9])
                        #[sha256,md5,sha1,timee,updatetime,family,platform,category,filesize,name]
                        sql = "INSERT INTO test1 (SHA256,MD5,SHA1,TIMEE,FTIME,FAMILY,PLATF,CATEG,LEN,VIRNAME,FLAG,PLATT) VALUES (?,?,?,?,?,?,?,?,?,?,1,0)"
                        c.executemany(sql, secondzip)
                        conn.commit()
                        print(i + "/" + j + "/" + k + "/" + l + "/" + m + "\n")
                       
    conn.close()


def main():
    conn = sqlite3.connect('data4.db')
    c = conn.cursor()
    cursor = c.execute("SELECT * FROM test1 ORDER BY FTIME")
    for row in cursor:
        print("NAME = ", row[0])
        print("NAME = ", row[1])
        print("NAME = ", row[2])
        print("NAME = ", row[3])
        print("NAME = ", row[4])
        print("NAME = ", row[5])
        print("NAME = ", row[6])
        print("NAME = ", row[7])
        print("NAME = ", row[8])
        print("NAME = ", row[9])
        print("NAME = ", row[10])
        print("NAME = ", row[11], "\n")
    cursor = c.execute("select count(*) from test1")
    print(cursor.fetchone()[0])
    conn.close()


if __name__ == "__main__":
    main()
