#!/usr/bin/env python3
# -*-coding: utf-8 -*-
'''
__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"
__contact__ = "liying_china@163.com"
'''


from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
#from flask_paginate import Pagination, get_page_args
import json
import pandas as pd
import sys
import imp
import re
import os
import string
from search import get_info_by_sha256 
from search import get_info_by_md5 
from search import get_info_all
from web_download import get_torrent_file
from web_download import get_tgz_file
from web_download import get_torrent_files
from web_download import get_tgz_files
from DGA_detection import LSTM_classifier, XGBoost_classifier, RF_classifier, SVM_classifier

HOST_IP = "0.0.0.0"
PORT = 5002
ROW_PER_PAGE = 20

imp.reload(sys)


app = Flask(__name__)

labels = None
contents = None
slabels = None
scontents = None
title = None
list_sha256 = []

list_info = []

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, str):
            return str(obj)


def get_page_info(list_info, offset=0, per_page=ROW_PER_PAGE):
    return list_info[offset: offset + per_page]

@app.route('/search_all', methods=['POST'])
def search_all():
    global list_sha256
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
    list_info = get_info_all(platform, category, family, scan_result, year, feature)

    # 3. Check match list 
    if not len(list_info):
        return render_template('error.html', \
                title = "没有找到符合条件的恶意代码样本",\
                scan_sha256 = "")
    list_sha256 = []
    for i in list_info:
        list_sha256.append(i["sha256"])

    print(len(list_sha256))
    print(list_sha256[0])

    print(len(list_sha256))
    # 4. Use list.html template to show search results
    return render_template('list.html', \
            list_info = list_info)

#    # 3. Pagination
#    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
#    total = len(list_info)
#    pagination_list_info = get_page_info(list_info, offset=offset, per_page=per_page)
#    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')
#
#    # 3. Use list.html template to show search results
#    return render_template('list.html', \
#            list_info = pagination_list_info,\
#            page=page,\
#            per_page=per_page,\
#            pagination=pagination) 


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


# Search SHA256
@app.route('/search_sha256', methods=['POST'])
def search_sha256():
    # 1. Get sha256
    sha256 = request.form['sha256']

    # 2. Validate sha256
    # 2.1 check length of sha256 string
    if len(sha256) != 64:
        return render_template('error.html', \
                title = "SHA256字符串不合法,长度不是64字符.",\
                scan_sha256 = sha256)
    # 2.2 Check hexdecimal characters
    if not all(x in string.hexdigits for x in str(sha256)):
        return render_template('error.html', \
                title = "SHA256字符串不合法,包含不合法的十六进制字符.",\
                scan_sha256 = sha256)

    # 3. Get json info
    dict_json = get_info_by_sha256(sha256)
    if not dict_json:
        return render_template('error.html', \
                title = "恶意代码样本库中没有找到样本信息",\
                scan_sha256 = sha256)

    title = "恶意代码样本信息"

    # 4. Get scan results
    if len(dict_json.keys()) == 2:
        scans = dict_json["results"]["scans"]
        md5 = dict_json["results"]["md5"]
    else:
        scans = dict_json['scans']
        md5 = dict_json["md5"]

    d = {}
    for key, value in scans.items():
        if value['detected']:
            d[key] = {'result':value['result'], 'version':value['version']}
        else:
            d[key] = {'result':"CLEAN", 'version':value['version']}
        #if not value['detected']: # Only show detected results
        #    continue 
        #d[key] = {'result':value['result'], 'version':value['version']}

    kav_result = d["Kaspersky"]["result"]
    if ":" in kav_result:
        kav_result = kav_result.split(":")[1]
    print(kav_result)
    # AdWare.MSIL.Ocna.aps
    list_kav = kav_result.split(".")
    category = list_kav[0]
    platform = list_kav[1]
    family = list_kav[2]

    #for key, value in d.items():
    #    print("{}: {}".format(key, value))

    return render_template('detail.html', \
            title = title,\
            scans = d,\
            scan_sha256 = sha256,\
            scan_md5 = md5,\
            platform = platform,\
            category = category,\
            family = family)


# Search MD5
@app.route('/search_md5', methods=['POST'])
def search_md5():
    # 1. Get MD5
    md5 = request.form['md5']

    # 2. Validate MD5
    # 2.1 check length of md5 string
    if len(md5) != 32:
        return render_template('error.html', \
                title = "MD5字符串不合法,长度不是32字符.",\
                scan_md5 = md5,\
                scan_sha256 = "")
    # 2.2 Check hexdecimal characters
    if not all(x in string.hexdigits for x in str(md5)):
        return render_template('error.html', \
                title = "MD5字符串不合法,包含不合法的十六进制字符.",\
                scan_md5 = md5,\
                scan_sha256 = "")

    # 4. Get json info
    dict_json = get_info_by_md5(md5)
    if not dict_json:
        return render_template('error.html', \
                title = "恶意代码样本库中没有找到样本信息",\
                scan_md5 = md5,\
                scan_sha256 = "")
    title = "恶意代码样本信息"

    # 5. Get scan results
    if len(dict_json.keys()) == 2:
        scans = dict_json["results"]["scans"]
        sha256 = dict_json["results"]["sha256"]
    else:
        scans = dict_json['scans']
        sha256 = dict_json["sha256"]
    d = {}
    for key, value in scans.items():
        if value['detected']:
            d[key] = {'result':value['result'], 'version':value['version']}
        else:
            d[key] = {'result':"CLEAN", 'version':value['version']}

        #if not value['detected']: # Only show detected results
        #    continue 
        #d[key] = {'result':value['result'], 'version':value['version']}

    kav_result = d["Kaspersky"]["result"]
    if ":" in kav_result:
        kav_result = kav_result.split(":")[1]
    print(kav_result)
    # AdWare.MSIL.Ocna.aps
    list_kav = kav_result.split(".")
    category = list_kav[0]
    platform = list_kav[1]
    family = list_kav[2]

    return render_template('detail.html', \
            title = title,\
            scans = d,\
            scan_md5 = md5,\
            scan_sha256 = sha256,\
            platform = platform,\
            category = category,\
            family = family)

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
def show_index():
    return render_template('index.html', element=1)

@app.route('/malware_url')
def show_malUrl():
    return render_template("malware_url_query.html", element=2)

@app.route('/malware_reuslt', methods=["POST"])
def detect_url():
    # 1. get url string
    url_str = request.form["url"].strip()
    # 2. validate string
    if url_str == '':
        return render_template("malware_url_result.html",
                           element=2, status=400, url=url_str,
                           message="域名不可为空!!")
    validate = re.match(r"^[A-Za-z0-9._\-]*$", url_str)
    if validate == None:
        return render_template("malware_url_result.html",
                               element=2, status=401, url=url_str,
                               message="域名格式不正确，请输入正确域名！！",
                               message2="（域名中只能包含下划线、短横线、点、字母、数字）")
    # 3. detect the str with models
    SVM = SVM_clf.predict_singleDN(model_folder, url_str)
    BRF = RF_clf.predict_singleDN(model_folder, url_str)
    XGBOOST = XGBoost_clf.predict_singleDN(model_folder, url_str)
    LSTM = LSTM_clf.predict_singleDN(url_str)

    base_result = {}
    result = -1
    base_result["SVM"] = [SVM[0], format(SVM[1], '.4f'), SVM[2]]
    base_result["B-RF"] = [BRF[0], format(BRF[1], '.4f'), BRF[2]]
    base_result["XGBOOST"] = [XGBOOST[0], format(XGBOOST[1], '.4f'), XGBOOST[2]]
    base_result["LSTM"] = [LSTM[0], format(LSTM[1], '.4f'), LSTM[2]]

    ## 4.p
    SVM = SVM if SVM[2] > 0.01 else (2, SVM[1], SVM[2])
    BRF = BRF if BRF[2] > 0.01 else (2, BRF[1], BRF[2])
    XGBOOST = XGBOOST if XGBOOST[2] > 0.01 else (2, XGBOOST[1], XGBOOST[2])
    LSTM = LSTM if LSTM[2] > 0.01 else (2, LSTM[1], LSTM[2])

    base_resultT = {"SVM": SVM, "B-RF": BRF, "XGBOOST": XGBOOST, "LSTM": LSTM}

    ## 5.result
    rs_list = []
    for item in base_resultT:
        rs_list.append(base_resultT[item][0])
    if len(set(rs_list)) == 1:
        if LSTM[0] != 2:
            result = LSTM[0]
            return render_template("malware_url_result.html", status=200, url=url_str, base_result=base_result,
                                   result=result, element=2)
        elif LSTM[0] == 2:  # 所有模型都表现很差
            sort_result = sorted(base_resultT.items(), key=lambda base_resultT: base_resultT[1][2], reverse=True)
            if (sort_result[0][1][2] <= 0.5):
                result = 2
            else:
                result = sort_result[0][1][0]
            return render_template("malware_url_result.html", status=200, url=url_str, base_result=base_result,
                                   result=result, element=2)

    new_result = {}
    for item in base_resultT:
        if base_resultT[item][0] != 2:
            new_result[item] = base_resultT[item]
    sort_result = sorted(new_result.items(), key=lambda new_result: new_result[1][2], reverse=True)
    if (sort_result[0][1][2] <= 0.5):
        result = 2
    else:
        result = sort_result[0][1][0]
    return render_template("malware_url_result.html", status=200, url=url_str, base_result=base_result,
                           result=result, element=2)


@app.route('/sha256/<sha256>')
def download_sha256(sha256):
    path = "../DATA/sha256/" + sha256[0] + '/' +  sha256[1] + '/' + sha256[2] + '/' + sha256[3] + '/' + sha256
    path = os.path.abspath(path)
    print(path)
    return send_file(path, as_attachment=True)

@app.route('/tgz/<sha256>')
def download_tgz(sha256):
    f_tgz = get_tgz_file(sha256)
    print("[Web] Get tgz file {}".format(f_tgz))
    return send_file(f_tgz, as_attachment=True)

@app.route('/torrent/<sha256>')
def download_torrent(sha256):
    f_torrent = get_torrent_file(sha256)
    print("[Web] Get torrent file {}".format(f_torrent))
    return send_file(f_torrent, as_attachment=True)

@app.route('/tgz_list/')
def download_tgz_list():
    global list_sha256
    print(list_sha256[0])
    f_tgz = get_tgz_files(list_sha256)
    print(f_tgz)
    print("[Web] Get tgz file {}".format(f_tgz))
    return send_file(f_tgz, as_attachment=True)

@app.route('/torrent_list/')
def download_torrent_list():
    global list_sha256
    f_torrent = get_torrent_files(list_sha256)
    print("[Web] Get torrent file {}".format(f_torrent))
    return send_file(f_torrent, as_attachment=True)

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


LSTM_model_add = r"./data/model/LSTM_model.json"
LSTM_model_weight = r"./data/model/LSTM_model.h5"

model_folder = r"./data/model"
XGBoost_clf = XGBoost_classifier()
XGBoost_clf.load(model_folder)
RF_clf = RF_classifier()
RF_clf.load(model_folder)
SVM_clf = SVM_classifier()
SVM_clf.load(model_folder)
LSTM_clf = LSTM_classifier()
LSTM_clf.load(LSTM_model_add, LSTM_model_weight)

if __name__ == '__main__':
    # debug=True, , threaded=True
    app.run(host=HOST_IP, port=PORT)

