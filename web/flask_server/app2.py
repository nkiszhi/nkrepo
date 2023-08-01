#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
Created on 2022/1/3 13:35

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2022 NKAMG"
__license__ = "GPL"
__contact__ = ""

Description:

"""

from flask import Flask, render_template, request, send_file, jsonify
import json
import sys
import imp
import re
import os
import string
import sqlite3
from configparser import ConfigParser
from dga_detection import MultiModelDetection
from search import get_info_by_sha256
from search import get_info_by_md5
from search import get_info_all
from web_download import get_torrent_file
from web_download import get_tgz_file
from web_download import get_torrent_files
from web_download import get_tgz_files

cp = ConfigParser()
cp.read('config.ini')
HOST_IP = cp.get('ini', 'ip')
PORT = int(cp.get('ini', 'port'))
ROW_PER_PAGE = int(cp.get('ini', 'row_per_page'))
detector = MultiModelDetection()

imp.reload(sys)
app = Flask(__name__)


from flask import Flask, jsonify, request, make_response, send_from_directory
import os
import sqlite3
import hashlib
import requests
import base64
# sslify = SSLify(app)
context = ("/home/nkamg/final.crt","/home/nkamg/example.key")
ROOTPATH = "./DATA/sha256/"
ROOT_PATH = "./tmp/"

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

def get_md5(file_path: str) -> str:
    return get_hash(file_path, hashlib.md5)

def get_sha1(file_path: str) -> str:
    return get_hash(file_path, hashlib.sha1)

@app.route('/1')
def helloworld():
    return "hello"

@app.route('/services/sampleLib/syncInfoList', methods=["POST"])
def third_post():
    my_json = request.get_json()
    mytype = my_json.get("platType")
    mysize = my_json.get("pageSize")
    mynum = my_json.get("pageNum")
    mycode = 200
    mymsg = "操作成功"
    conn = sqlite3.connect('data4.db')
    c = conn.cursor()
    cursor = c.execute("SELECT * FROM test1 ORDER BY TIMEE LIMIT ?,?",(mysize*(mynum-1),mysize))
    result = []
    for row in cursor:
        dictl = {
            "name": row[0],
            "md5": row[1],
            "sha1": row[2],
            "sha256": row[0],
            "virusFlag": 1,
            "sysTypeCode": "",
            "virusTypeCode": "",
            "virusTypeName": "",
            "fileType": "",
            "families": row[6],
            "virusName": row[11],
            "platType": mytype,
            "platform": row[8]
        }
        result.append(dictl)
    cursor = c.execute("select count(*) from test1")
    n = cursor.fetchone()[0]
    conn.close()
    mydata={"pageSize":mysize,"pageNum":mynum,"total":n,"infoList":result}
    return jsonify(code=mycode, msg=mymsg, data=mydata)

@app.route('/services/sampleLib/getSampleFile', methods=["POST"])
def get_samplefile():
    my_json = request.get_json()
    print(my_json)
    mysha256 = my_json.get("sha256")
    filetag = ROOT_PATH + mysha256 + ".mal"
    mycode = 500
    mymsg = "操作失败"
    mydata = {"downLink":"","fileData":""}
    if os.path.isfile(filetag):
        mycode = 200
        mymsg = "操作成功"
        mydata = {"downLink":"https://192.168.7.44:5000/download?fileName="+mysha256+".mal","fileData":""}
    return jsonify(code=mycode, msg=mymsg, data=mydata)

@app.route("/download", methods=['GET'])
def download_file():
    get_data = request.args.to_dict()
    file_path = get_data.get('fileName')
 
    response = make_response(
        send_from_directory(ROOT_PATH,file_path,as_attachment=True))
    response.headers["Content-Disposition"] = "attachment; filename={}".format(
        file_path.encode().decode('latin-1'))
    return response

@app.route('/services/sampleLib/sandboxImport', methods=["POST"])
def import_sandbox():
    print("try import_sandbox")
    my_json = request.get_json()
    mysha256 = my_json.get("fileSha256")
    url = my_json.get("importUrl")
    myinfo = my_json.get("params")
    myinfo = base64.b64decode(myinfo)
    myinfo = str(myinfo, 'utf-8')
    plist = myinfo.split(',')
    pdict = {}
    for i in range(len(plist)):
        pdict.update({plist[i].split('=')[0]: plist[i][plist[i].find('=')+1:len(plist[i])]})
    if os.path.exists(ROOT_PATH + mysha256 + ".sgcceyyb"):
        fields = {
            "encFlag": 1,
            "key": pdict["key"],
            "rescan": pdict["rescan"],
            "platform": "",
            "timeout": 60
        }
        files = {
            'file': (mysha256, open(ROOT_PATH + mysha256 + ".sgcceyyb", 'rb'))
        }
        response = requests.post(url, data=fields, files=files)
        if response.status_code != 200:
            print("submit {}, fail {}, {}".format(mysha256, response.status_code, response.content))
            cdict = str(response.content, 'utf-8')
            cdict = eval(cdict)
            code = cdict["status_code"]
            msg = cdict["msg"]
        else:
            cdict = str(response.content, 'utf-8')
            cdict = eval(cdict)
            code = cdict["status_code"]
            msg = cdict["msg"]
            print("submit {} ok: {}".format(mysha256, response.content))
    else:
        code = 500
        msg = " 没有该样本"
        print("no sample {}".format(mysha256))
    return jsonify(code=code, msg=msg)

@app.route('/services/sampleLib/testImport', methods=["POST"])
def import_test():
    my_json = request.get_json()
    mysha256 = my_json.get("fileSha256")
    url = my_json.get("importUrl")
    myinfo = my_json.get("params")
    myinfo = base64.b64decode(myinfo)
    myinfo = str(myinfo, 'utf-8')
    plist = myinfo.split(',')
    pdict = {}
    for i in range(len(plist)):
        pdict.update({plist[i].split('=')[0]: plist[i][plist[i].find('=')+1:len(plist[i])]})
    if os.path.exists(ROOT_PATH + mysha256 + ".sgcceyyb"):
        fields = {
            "submitKey": pdict["submitKey"]
        }
        files = {
            'file': (mysha256, open(ROOT_PATH + mysha256 + ".sgcceyyb", 'rb'))
        }
        response = requests.post(url, data=fields, files=files)
        if response.status_code != 200:
            print("submit {}, fail {}, {}".format(mysha256, response.status_code, response.content))
            cdict = str(response.content, 'utf-8')
            cdict = eval(cdict)
            code = cdict["status_code"]
            msg = cdict["msg"]
        else:
            code = 200
            msg = "操作成功"
            print("submit {} ok: {}".format(mysha256, response.content))
    else:
        code = 500
        msg = " 没有该样本"
        print("no sample {}".format(mysha256))
    return jsonify(code=code, msg=msg)

    

if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, threaded=True)
