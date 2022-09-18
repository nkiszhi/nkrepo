#!/usr/bin/env python3
# -*-coding: utf-8 -*-

from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
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

app = Flask(__name__)

CORS().init_app(app)
detector = MultiModelDetection()

list_info = []

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, str):
            return str(obj)

@app.route('/malurl_query', methods=["POST", "GET"])
def malurl_page():
    return render_template("malware_url_query.html")

@app.route('/malurl_result', methods=["POST"])
def detect_url():
    # 1. get url string
    url_str = request.form["url"].strip()
    # 2. validate string
    if url_str == '':
        return render_template("malware_url_result.html",
                           status=400, url=url_str,
                           message="域名不可为空!!")
    validate = re.match(r"^[A-Za-z0-9._\-]*$", url_str)
    if validate == None:
        return render_template("malware_url_result.html",
                               status=401, url=url_str,
                               message="域名格式不正确，域名中只能包含下划线、短横线、点、字母、数字，请输入正确域名！")
    results = detector.multi_predict_single_dname(url_str)
    return render_template("malware_url_result.html", status=200, url=url_str, base_result=results[0],
                           result=results[1])

# 样本查询页面
@app.route('/malsample_search')
def malsample_search_page():
    return render_template('malicious_sample_search.html')

# 综合查询
@app.route('/search_all', methods=['POST'])
def search_all():
    conn = sqlite3.connect('data4.db')
    c = conn.cursor()
    global list_sha256
    # 1. Get search parameters
    platform = request.form['platform']
    category = request.form['category']
    family = request.form['family']
    year = request.form['year']

    # 2. Get matched sha256 list
    strsql = "SELECT * FROM test1 WHERE "
    listsql = []
    if not platform == "Any":
        strsql = strsql + "PLATF =? AND "
        listsql.append(platform)
    if not category == "Any":
        strsql = strsql + "CATEG =? AND "
        listsql.append(category)
    if not family == "Any":
        strsql = strsql + "FAMILY =? AND "
        listsql.append(family)
    if not year == "Any":
        yyear = "%{}%".format(year)
        strsql = strsql + "FTIME LIKE ? AND "
        listsql.append(yyear)
    strsql = strsql.strip(" AND ")
    print("strsql:"+strsql)
    cursor = c.execute(strsql,tuple(listsql))
    flag = 0
    list_sha256 = []
    list_info = []
    for row in cursor:
        flag = 1
        list_sha256.append(row[0])
        list_info.append({"sha256":row[0], "md5":row[1], "year":row[4]})

    # list_info = get_info_all(platform, category, family, scan_result, year, feature)

    # 3. Check match list
    if not flag:
        return render_template('error.html', \
                title = "没有找到符合条件的恶意代码样本",\
                scan_sha256 = "")

    print(len(list_sha256))
    print(list_sha256[0])

    # 4. Use list.html template to show search results
    return render_template('list.html',
                           list_info = list_info)

# SHA256搜索
@app.route('/search_sha256', methods=['POST'])
def search_sha256():
    conn = sqlite3.connect('data4.db')
    c = conn.cursor()
    # 1. Get sha256
    sha256 = request.form['sha256']
    
    # 2. Validate sha256
    # 2.1 check length of sha256 string
    if len(sha256) != 64:
        return render_template('error.html',
                               title="SHA256字符串不合法，长度不是64字符！",
                               scan_sha256=sha256)
    # 2.2 Check hexdecimal characters
    if not all(x in string.hexdigits for x in str(sha256)):
        return render_template('error.html',
                               title="SHA256字符串不合法，包含不合法的十六进制字符！",
                               scan_sha256=sha256)
    cursor = c.execute("SELECT * FROM test1 WHERE SHA256 =?",(sha256,))
    flag = 0
    for row in cursor:
        flag = 1
        category = row[9]
        platform = row[8]
        family = row[6]
        md5 = row[1]

    # 3. Get json info
    dict_json = get_info_by_sha256(sha256)
    if not flag:
        return render_template('error.html',
                               title="恶意代码样本库中没有找到样本信息",
                               scan_sha256=sha256)

    title = "恶意代码样本信息"

    # 4. Get scan results
    if len(dict_json.keys()) == 2:
        scans = dict_json["results"]["scans"]
    else:
        scans = dict_json['scans']

    d = {}
    for key, value in scans.items():
        if value['detected']:
            d[key] = {'result': value['result'], 'version': value['version']}
        else:
            d[key] = {'result': "CLEAN", 'version': value['version']}

    kav_result = d["Kaspersky"]["result"]
    if ":" in kav_result:
        kav_result = kav_result.split(":")[1]
    print(kav_result)
    # AdWare.MSIL.Ocna.aps
    #list_kav = kav_result.split(".")
    #category = list_kav[0]
    #platform = list_kav[1]
    #family = list_kav[2]

    return render_template('detail.html',
                           title=title,
                           scans=d,
                           scan_sha256=sha256,
                           scan_md5=md5,
                           platform=platform,
                           category=category,
                           family=family)


# MD5搜索
@app.route('/search_md5', methods=['POST'])
def search_md5():
    conn = sqlite3.connect('data4.db')
    c = conn.cursor()
    # 1. Get MD5
    md5 = request.form['md5']

    # 2. Validate MD5
    # 2.1 check length of md5 string
    if len(md5) != 32:
        return render_template('error.html',
                               title="MD5字符串不合法，长度不是32字符！",
                               scan_md5=md5,
                               scan_sha256="")
    # 2.2 Check hexdecimal characters
    if not all(x in string.hexdigits for x in str(md5)):
        return render_template('error.html',
                               title="MD5字符串不合法，包含不合法的十六进制字符！",
                               scan_md5=md5,
                               scan_sha256="")

    cursor = c.execute("SELECT * FROM test1 WHERE MD5 =?",(md5,))
    flag = 0
    for row in cursor:
        flag = 1
        category = row[9]
        platform = row[8]
        family = row[6]
        sha256 = row[0]

    # 4. Get json info
    dict_json = get_info_by_sha256(sha256)
    print(dict_json)
    if not flag:
        return render_template('error.html',
                               title="恶意代码样本库中没有找到样本信息",
                               scan_md5=md5,
                               scan_sha256="")
    title = "恶意代码样本信息"

    # 5. Get scan results
    if len(dict_json.keys()) == 2:
        scans = dict_json["results"]["scans"]
    else:
        scans = dict_json['scans']

    d = {}
    for key, value in scans.items():
        if value['detected']:
            d[key] = {'result': value['result'], 'version': value['version']}
        else:
            d[key] = {'result': "CLEAN", 'version': value['version']}

    kav_result = d["Kaspersky"]["result"]
    if ":" in kav_result:
        kav_result = kav_result.split(":")[1]
    print(kav_result)
    # AdWare.MSIL.Ocna.aps
    # list_kav = kav_result.split(".")
    # category = list_kav[0]
    # platform = list_kav[1]
    # family = list_kav[2]

    return render_template('detail.html',
                           title=title,
                           scans=d,
                           scan_md5=md5,
                           scan_sha256=sha256,
                           platform=platform,
                           category=category,
                           family=family)

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


@app.route('/api/<path>')
def hello_world(path):  # put application's code here
    filepath = './static/data/' + path + '.json'
    if not os.path.exists(filepath):
        return jsonify({
            'message': '读取文件内容失败, 文件资源不存在',
            'status': 404
        })
    return send_file(filepath)

if __name__ == '__main__':
    app.run(host="192.168.7.44", port=5004, threaded=True)
