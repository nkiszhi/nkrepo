#coding:utf-8
import os,math,datetime
from flask import Flask,render_template,request,send_from_directory
import hashlib,pefile,peutils,re
from backstage.WebTotal import Web
app = Flask(__name__)

# 后台功能载入
web = Web()

# 主网页部分
@app.route('/', methods=['GET'])
def loadIndex() :
    # 更新网页
    web.update()
    
    # 返回主网页
    return web.getMainWeb()

# 搜索部分
@app.route('/search', methods=['GET'])
def loadSearch() :
    for key, value in request.args.items() : 
        if key == "Sha256Search" : 
            return web.getSha256Result(value)
        if key == "FamilySearch" : 
            if not "Page" in request.args.keys() : break
            return web.getFamilySearchResult(value, request.args["Page"])
        if key == "TimeSearch" : 
            if not "Page" in request.args.keys() : break
            return web.getTimeResult(value, request.args["Page"])
        if key == "HybirdSearch" :
            if not "Page" in request.args.keys() : break
            return web.getHybirdSearchResult(value, request.args["Page"])
    return "{}"

@app.route('/download/<sha256>',methods=['GET'])
def downloadfile(sha256):
    if not re.search('[0-9a-zA-Z]{64}', sha256) : return "no such file"
    if os.path.exists("../NKVSDATA/"+sha256[0]+'/'+sha256[1]+'/'+sha256[2]+'/'+sha256[3]+'/'+sha256+".vs"):
        return send_from_directory("../NKVSDATA/"+sha256[0]+'/'+sha256[1]+'/'+sha256[2]+'/'+sha256[3]+'/',  sha256+".vs"  ,as_attachment=True)
    elif os.path.exists("../NKAZDATA/"+sha256[0]+'/'+sha256[1]+'/'+sha256[2]+'/'+sha256[3]+'/'+sha256+".az"):
        return send_from_directory("../NKAZDATA/"+sha256[0]+'/'+sha256[1]+'/'+sha256[2]+'/'+sha256[3]+'/',  sha256+".az"  ,as_attachment=True)
    else:
        return "no such file"

if __name__=='__main__':
    app.run(host = '0.0.0.0' , port = 5000, debug = True)