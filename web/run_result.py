#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Author: liying
@Email: liying_china@163.com
@File: run.py
@Create Time: 2020/7/10 上午9:53
"""


from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import pandas as pd
import sys
import os
from search import get_sha256_info 
import sys
reload(sys)
sys.setdefaultencoding('utf8') 
#sys.setdefaultencoding('utf8')
#HOST_IP = "60.205.204.64"
HOST_IP = "0.0.0.0"
PORT = 5000

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


@app.route('/')
def graph_ip():
    return render_template('/graph_result.html')

@app.route('/detail')
def detail():
    print(scontents)
    return render_template('detail.html', labels=labels, \
            contents=contents,title=title,slabels=slabels,scontents=scontents,stitle=stitle,atitle=atitle,alabels=alabels,acontents=acontents,ititle=ititle,ilabels=ilabels,icontents=icontents,icontents_=icontents_,etitle=etitle,elabels=elabels,econtents=econtents)

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/search_data', methods=['post', 'get'])
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
    S = request.get_data()
    # get annual sales rank
    print(S)
    T = bytes.decode(S)
    print(999999)
    print(T)
    '''
    str_cmd = "python2 search.py -s {}".format(T)
    filetype =  os.popen(str_cmd).read().strip().split("?")
    print(filetype)
    '''
    title,labels,contents,slabels,scontents,stitle,atitle,alabels,acontents,ititle,ilabels,icontents,icontents_,etitle,elabels,econtents = get_sha256_info(T)
    #labels=[]
    #contents=[]
    #title = ""
    print(ilabels)
    print(icontents)
    print(66666) 
    print(econtents)
    #return redirect(url_for('detail',labels=labels,content=content,jlabels=jlabels,jcontent=jcontent))
    return jsonify({title:title,"labels":labels,"contents":contents,"slabels":slabels,"scontents":scontents,"stitle":stitle,"atitle":atitle,"alabels":alabels,"acontents":acontents,"ititle":ititle,"ilabels":ilabels,"icontents":icontents,"icontents_":icontents_,"etitle":etitle,"elabels":elabels,"econtents":econtents})




'''
def map_rate(x, to_min, to_max, max_num, min_num):
    """
    将指定的访问次数映射到区间范围，作为点的大小
    :param x:
    :return:
    """
    return round(to_min + ((to_max - to_min) / (max_num - min_num)) * (x - min_num), 1)


@app.route('/get_knn')
def get_knn_data():
    """

    :return:
    """
    a = os.path.dirname(__file__)
    safe = pd.read_csv(a + '/data/KNN_safe.csv')
    dangerous = pd.read_csv(a + '/data/KNN_dangerous.csv')
    lss = []
    tmp = {}
    for i in range(len(safe)):
        dic = []
        dic.append(str(safe['0'][i]))
        dic.append(str(safe['1'][i]))
        dic.append(str(safe['2'][i]))
        dic.append(str(safe['IP'][i]))
        lss.append(dic)
    tmp['safe'] = lss

    lssd = []
    for i in range(len(dangerous)):
        dic = []
        dic.append(str(dangerous['0'][i]))
        dic.append(str(dangerous['1'][i]))
        dic.append(str(dangerous['2'][i]))
        dic.append(str(dangerous['IP'][i]))
        lssd.append(dic)
    tmp['dangerous'] = lssd

    sus = dangerous['IP'][:]
    top10 = sus.value_counts()[:10]
    lse = []
    sum_n = 0
    for i in range(len(top10)):
        sum_n += top10[i]
        dic = {"name": str(top10.index[i]), 'value': str(top10[i])}
        lse.append(dic)
    other_sum = len(lssd) - sum_n
    lse.append({"name": "其他IP", 'value': str(other_sum)})
    tmp['knn_bing'] = lse

    return jsonify(tmp)


@app.route('/get_pca')
def get_pca_data():
    """

    :return:
    """
    a = os.path.dirname(__file__)
    safe = pd.read_csv(a + '/data/PCA_safe.csv')
    dangerous = pd.read_csv(a + '/data/PCA_dangerous.csv')
    lss = []
    tmp = {}
    for i in range(len(safe)):
        dic = []
        dic.append(str(safe['0'][i]))
        dic.append(str(safe['1'][i]))
        dic.append(str(safe['2'][i]))
        dic.append(str(safe['IP'][i]))
        lss.append(dic)
    tmp['safe'] = lss

    lssd = []
    for i in range(len(dangerous)):
        dic = []
        dic.append(str(dangerous['0'][i]))
        dic.append(str(dangerous['1'][i]))
        dic.append(str(dangerous['2'][i]))
        dic.append(str(dangerous['IP'][i]))
        lssd.append(dic)
    tmp['dangerous'] = lssd

    sus = dangerous['IP'][:]
    top10 = sus.value_counts()[:10]
    lse = []
    sum_n = 0
    for i in range(len(top10)):
        sum_n += top10[i]
        dic = {"name": str(top10.index[i]), 'value': str(top10[i])}
        lse.append(dic)
    other_sum = len(lssd) - sum_n
    lse.append({"name": "其他IP", 'value': str(other_sum)})
    tmp['pca_bing'] = lse

    return jsonify(tmp)


@app.route('/get_lof')
def get_lof_data():
    """

    :return:
    """
    a = os.path.dirname(__file__)
    safe = pd.read_csv(a + '/data/LOF_safe.csv')
    dangerous = pd.read_csv(a + '/data/LOF_dangerous.csv')
    lss = []
    tmp = {}
    for i in range(len(safe)):
        dic = []
        dic.append(str(safe['0'][i]))
        dic.append(str(safe['1'][i]))
        dic.append(str(safe['2'][i]))
        dic.append(str(safe['IP'][i]))
        lss.append(dic)
    tmp['safe'] = lss

    lssd = []
    for i in range(len(dangerous)):
        dic = []
        dic.append(str(dangerous['0'][i]))
        dic.append(str(dangerous['1'][i]))
        dic.append(str(dangerous['2'][i]))
        dic.append(str(dangerous['IP'][i]))
        lssd.append(dic)
    tmp['dangerous'] = lssd

    sus = dangerous['IP'][:]
    top10 = sus.value_counts()[:10]
    lse = []
    sum_n = 0
    for i in range(len(top10)):
        sum_n += top10[i]
        dic = {"name": str(top10.index[i]), 'value': str(top10[i])}
        lse.append(dic)
    other_sum = len(lssd) - sum_n
    lse.append({"name": "其他IP", 'value': str(other_sum)})
    tmp['pca_bing'] = lse

    return jsonify(tmp)


@app.route('/get_vae')
def get_vae_data():
    """

    :return:
    """
    a = os.path.dirname(__file__)
    safe = pd.read_csv(a + '/data/VAE_safe.csv')
    dangerous = pd.read_csv(a + '/data/VAE_dangerous.csv')
    lss = []
    tmp = {}
    for i in range(len(safe)):
        dic = []
        dic.append(str(safe['0'][i]))
        dic.append(str(safe['1'][i]))
        dic.append(str(safe['2'][i]))
        dic.append(str(safe['IP'][i]))
        lss.append(dic)
    tmp['safe'] = lss

    lssd = []
    for i in range(len(dangerous)):
        dic = []
        dic.append(str(dangerous['0'][i]))
        dic.append(str(dangerous['1'][i]))
        dic.append(str(dangerous['2'][i]))
        dic.append(str(dangerous['IP'][i]))
        lssd.append(dic)
    tmp['dangerous'] = lssd

    sus = dangerous['IP'][:]
    top10 = sus.value_counts()[:10]
    lse = []
    sum_n = 0
    for i in range(len(top10)):
        sum_n += top10[i]
        dic = {"name": str(top10.index[i]), 'value': str(top10[i])}
        lse.append(dic)
    other_sum = len(lssd) - sum_n
    lse.append({"name": "其他IP", 'value': str(other_sum)})
    tmp['pca_bing'] = lse

    return jsonify(tmp)


@app.route('/get_scan_230')
def get_scan_data():
    """
        427个点
    10.12.1.230:235
    10.12.80.87:155
    :return:
    """
    a = os.path.dirname(__file__)
    file_dir = a + '/data/scan_1_230.csv'
    name_df = pd.read_csv(file_dir)
    src_ip = name_df['src_ip']  #.groupby('src_ip')
    dst_ip = name_df['dst_ip']

    count_data = src_ip.value_counts()[:5]
    count_data = count_data.index  # 联系数量排名前5的ip
    dst_count_list = []  # 前5的ip涉及到的所有点
    for i in range(len(src_ip)):
        if src_ip[i] in count_data:
            dst_count_list.append(dst_ip[i])
    dst_count_list.append(count_data[0])
    dst_count_list.append(count_data[1])
    dst_count_list.append(count_data[2])
    dst_count_list.append(count_data[3])
    dst_count_list.append(count_data[4])
    all_ip = list(set(dst_count_list))

    lss = {}
    ls = []
    for i in range(len(all_ip)):
        dic = {}
        dic['name'] = str(all_ip[i])
        if str(all_ip[i]) == '10.12.1.230':
            dic['category'] = '10.12.1.230(可疑IP)'
            dic['symbolSize'] = '30'
        elif str(all_ip[i]) in count_data:
            dic['category'] = str(all_ip[i])
            dic['symbolSize'] = '25'
        else:
            dic['category'] = 'other'
            dic['symbolSize'] = '10'
        ls.append(dic)
    lss['key'] = ls
    li = []
    for i in range(len(src_ip.values)):
        if str(src_ip[i]) in count_data:
            dic = {}
            dic['source'] = str(src_ip[i])
            dic['target'] = str(dst_ip[i])
            li.append(dic)
    lss['link'] = li

    lsc = []
    for i in count_data:
        dic = {}
        if str(i) == '10.12.1.230':
            dic['name'] = str(i) + '(可疑IP)'
        else:
            dic['name'] = str(i)
        lsc.append(dic)
    lsc.append({'name': 'other'})
    lss['cat'] = lsc
    return jsonify(lss)


@app.route('/get_scan_37')
def get_scan_data_37():
    """
    389个点
    10.12.1.37:194
    10.12.80.87:160
    :return:
    """
    a = os.path.dirname(__file__)
    file_dir = a + '/data/scan_1_37.csv'
    name_df = pd.read_csv(file_dir)
    src_ip = name_df['src_ip']  # .groupby('src_ip')
    dst_ip = name_df['dst_ip']

    count_data = src_ip.value_counts()[:5]
    count_data = count_data.index  # 联系数量排名前5的ip
    dst_count_list = []  # 前5的ip涉及到的所有点
    for i in range(len(src_ip)):
        if src_ip[i] in count_data:
            dst_count_list.append(dst_ip[i])
    dst_count_list.append(count_data[0])
    dst_count_list.append(count_data[1])
    dst_count_list.append(count_data[2])
    dst_count_list.append(count_data[3])
    dst_count_list.append(count_data[4])
    all_ip = list(set(dst_count_list))

    lss = {}
    ls = []
    for i in range(len(all_ip)):
        dic = {}
        dic['name'] = str(all_ip[i])
        if str(all_ip[i]) == '10.12.1.37':
            dic['category'] = '10.12.1.37(可疑IP)'
            dic['symbolSize'] = '30'
        elif str(all_ip[i]) in count_data:
            dic['category'] = str(all_ip[i])
            dic['symbolSize'] = '25'
        else:
            dic['category'] = 'other'
            dic['symbolSize'] = '10'
        ls.append(dic)
    lss['key'] = ls
    li = []
    for i in range(len(src_ip.values)):
        if str(src_ip[i]) in count_data:
            dic = {}
            dic['source'] = str(src_ip[i])
            dic['target'] = str(dst_ip[i])
            li.append(dic)
    lss['link'] = li

    lsc = []
    for i in count_data:
        dic = {}
        if str(i) == '10.12.1.37':
            dic['name'] = str(i) + '(可疑IP)'
        else:
            dic['name'] = str(i)
        lsc.append(dic)
    lsc.append({'name': 'other'})
    lss['cat'] = lsc
    return jsonify(lss)
    '''
if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, debug=True)
