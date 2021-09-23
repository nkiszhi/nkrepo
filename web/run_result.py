#!/usr/bin/env python3
# -*-coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"

from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
import json
import pandas as pd
import sys
import imp
import os
from search import get_json_info 

HOST_IP = "0.0.0.0"
PORT = 5000

imp.reload(sys)


app = Flask(__name__)

labels = None
contents = None
slabels = None
scontents = None
title = None

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, str):
            return str(obj)


#@app.route('/')
#def graph_ip():
#    return render_template('/graph_result.html')

@app.route('/detail')
def detail():
    scans = dict_json['scans']
    sha256 = dict_json['sha256']
    d = {}
    for key, value in scans.items():
        if value['detected']:
            d[key] = {'result':value['result'], 'version':value['version']}
        else:
            d[key] = {'result':"CLEAN", 'version':value['version']}

    for key, value in d.items():
        print("{}: {}".format(key, value))

    return render_template('detail.html', \
            title = title,\
            scans = d,\
            scan_sha256 = sha256)

@app.route('/')
def search():
    return render_template('search.html')

@app.route('/sha256/<sha256>')
def download_sha256(sha256):
    path = "../DATA/" + sha256[0] + '/' +  sha256[1] + '/' + sha256[2] + '/' + sha256[3] + '/' + sha256
    path = os.path.abspath(path)
    print(path)
    return send_file(path, as_attachment=True)

@app.route('/tgz/<tgz>')
def download_tgz(f_tgz):
    pass

@app.route('/torrent/<torrent>')
def download_torrent(f_torrent):
    pass

@app.route('/search_data', methods=['POST', 'GET'])
def search_data():
    global labels
    global contents
    global slabels
    global scontents
    global title
    global stitle
    global atitle
    global alabels
    global acontents
    global ititle
    global ilabels
    global icontents
    global icontents_
    global etitle
    global elabels
    global econtents
    global dict_json
    # 1. Get sha256 value for search
    sha256 = request.get_data()
    sha256 = bytes.decode(sha256)
    print("[i] Get SHA256: {}".format(sha256))
    #print(sha256)

    # 2. Get json info
    dict_json = get_json_info(sha256)
    if dict_json:
        title = "恶意代码样本信息"
    else:
        title = "恶意代码样本库中没有找到样本信息"

    # 3. Check 

    return jsonify({ title:title })



    #'''
    #str_cmd = "python2 search.py -s {}".format(T)
    #filetype =  os.popen(str_cmd).read().strip().split("?")
    #print(filetype)
    #'''

    #title,labels,contents,slabels,scontents,stitle,atitle,alabels,acontents,ititle,ilabels,icontents,icontents_,etitle,elabels,econtents =\
    #        get_sha256_info(T)
    ##labels=[]
    ##contents=[]
    ##title = ""
    #print(ilabels)
    #print(icontents)
    #print(66666) 
    #print(econtents)
    ##return redirect(url_for('detail',labels=labels,content=content,jlabels=jlabels,jcontent=jcontent))
    #return jsonify({title:title,\
    #        "labels":labels,\
    #        "contents":contents,\
    #        "slabels":slabels,\
    #        "scontents":scontents,\
    #        "stitle":stitle,\
    #        "atitle":atitle,\
    #        "alabels":alabels,\
    #        "acontents":acontents,\
    #        "ititle":ititle,\
    #        "ilabels":ilabels,\
    #        "icontents":icontents,\
    #        "icontents_":icontents_,\
    #        "etitle":etitle,\
    #        "elabels":elabels,\
    #        "econtents":econtents})


if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, debug=True)
