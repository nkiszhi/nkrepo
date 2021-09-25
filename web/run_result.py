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
from search import get_info_by_sha256 
from search import get_info_by_md5 
from search import get_info_all

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

@app.route('/search_all', methods=['POST'])
def search_all():
    # 0. test form data
    #print('search all\n\n')
    #for k, v in request.form.items():
    #    print("{}: {}".format(k, v))
    #print('search all\n\n')

    # 1. Get search parameters
    platform = request.form['platform']
    category = request.form['category']
    family = request.form['family']
    scan_result = request.form['scan_result']
    year = request.form['year']
    feature = request.form['feature']

    # 2. Get matched sha256 list
    list_sha256 = get_info_all(platform, category, family, scan_result, year, feature)

    for i in list_sha256:
        print(i)

    return render_template('blank.html', \
            list_sha256 = list_sha256) 


# Click sha256 to see detail information
@app.route('/show_sha256/<sha256>', methods=['POST'])
def show_info_by_sha256(sha256):
    # 1. Get json info
    print("show_info_by_sha256: {}".format(sha256))
    dict_json = get_info_by_sha256(sha256)
    if dict_json:
        title = "恶意代码样本信息"
    else:
        title = "恶意代码样本库中没有找到样本信息"

    # 2. Get scan results
    scans = dict_json['scans']
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

# SHA256 search
@app.route('/search_sha256', methods=['POST'])
def search_sha256():
    #for k, v in request.form.items():
    #    print("{}: {}".format(k, v))
    #print('search_sha256\n\n')
   
    # 1. Get sha256
    sha256 = request.form['sha256']

    # 2. Validate sha256
    # TODO

    # 3. Get json info
    dict_json = get_info_by_sha256(sha256)
    if dict_json:
        title = "恶意代码样本信息"
    else:
        title = "恶意代码样本库中没有找到样本信息"

    # 4. Get scan results
    scans = dict_json['scans']
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

# MD5 search
@app.route('/search_md5', methods=['POST'])
def search_md5():
    #for k, v in request.form.items():
    #    print("{}: {}".format(k, v))
    #print('search_sha256\n\n')
   
    # 1. Get MD5
    md5 = request.form['md5']

    # 2. Validate MD5
    # TODO

    # 3. Get sha256 value
    sha256 = get_sha256(md5)

    # 4. Get json info
    dict_json = get_info_by_sha256(sha256)
    if dict_json:
        title = "恶意代码样本信息"
    else:
        title = "恶意代码样本库中没有找到样本信息"

    # 5. Get scan results
    scans = dict_json['scans']
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
    path = "../DATA/sha256/" + sha256[0] + '/' +  sha256[1] + '/' + sha256[2] + '/' + sha256[3] + '/' + sha256
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

if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, debug=True)

