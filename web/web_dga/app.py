#!/usr/bin/env python3
# -*-coding: utf-8 -*-

from flask import Flask, render_template, request,make_response,redirect, url_for, session,request  
import json
import sys
import importlib
import re
from configparser import ConfigParser
from dga_detection import MultiModelDetection
from web_mysql import Databaseoperation
import pymysql
import os
import hashlib
from flask_login import LoginManager, login_user, logout_user, login_required, current_user  
from flask_sqlalchemy import SQLAlchemy  
from werkzeug.security import generate_password_hash, check_password_hash  
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from file_detect import EXEDetection
from web_user import db, User
cp = ConfigParser()
cp.read('config.ini')
HOST_IP = cp.get('ini', 'ip')
PORT = int(cp.get('ini', 'port'))
urldetector = MultiModelDetection()
querier =  Databaseoperation()



app = Flask(__name__)
app.config['SECRET_KEY'] = cp.get('mysql', 'SECRET_KEY')  
app.config['SQLALCHEMY_DATABASE_URI'] = cp.get('mysql', 'SQLALCHEMY_DATABASE_URI')  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = cp.getboolean('mysql', 'SQLALCHEMY_TRACK_MODIFICATIONS')  

db.init_app(app)  
login_manager = LoginManager()  
login_manager.init_app(app)  
login_manager.login_view = 'login'


@login_manager.user_loader  
def load_user(user_id):  
    return User.query.get(int(user_id))

def protected_route(func):  
    def wrapper(*args, **kwargs):  
        if 'logged_in' not in session or 'username' not in session:  
            return redirect(url_for('login'))
        return func(*args, **kwargs)  
    wrapper.endpoint = func.__name__
    return wrapper

    
@app.route('/login', methods=['GET', 'POST'])  
def login():  
        error=None
        if request.method == 'POST': 
          username = request.form.get('username')  
          password = request.form.get('password')  
          p_hash=generate_password_hash(password)#输出密码的哈希值
          print(username,password,p_hash)
          user = User.query.filter_by(username=username).first()  
          print(user)
          
          if user and user.check_password(password):  
            print("验证通过")
            login_user(user)  
            session['logged_in'] = True  
            session['username'] = username  
            print("已登录")
            next_page = request.args.get('next')  
            return redirect(next_page or url_for('home'))
          else:
            print(error)  
            error="用户名或密码错误"
            print(error)  
            # 登录失败，显示错误消息  
        return render_template('login.html', error=error) 

@app.route('/logout')  
def logout():  
    print("已注销")
    logout_user()  
    return redirect(url_for('login'))


#主页 
@app.route('/',endpoint='home',methods=['GET'])
def index():
    return render_template('malware_url_query.html')

  
@app.route('/statistics_url',endpoint='statistics_url')
#domain数据展示
def statistics_url ():
    return render_template('statistics_url.html')

@app.route('/statistics_file',endpoint='statistics_file')
#samples数据展示
def statistics_file():
    return render_template('statistics_file.html')
    
#@文件检测    
@app.route('/file',endpoint='malware_file_query',methods=['GET'])  
def show_indexs():  
    return render_template('malware_file_query.html')  
  
@app.route('/upload_material', methods=['GET', 'POST']) 
def uploadMaterialsToRgw():
    f = request.files['materials']
    print("上传的文件的名字是："+f.filename)
    str_name = f.filename

    if str_name == '':
      return render_template("malware_file_result.html",status=401,message="检测文件不可为空!!")        		
    
    #保存文件
    #localPath保存文件到磁盘的指定位置
    f.save('./web_file/%s'%(f.filename))
    file_size = os.path.getsize('./web_file/%s'%(f.filename))
    file_sizes = file_size / 1024
    file_size=str(file_size)
    file_sizes = format(file_sizes,'.2f')
    file_sizes = file_sizes +"KB"
    file_size = "("+file_size+" "+"bytes"+")"
    file_sizes = file_sizes+file_size 
    file_path = '/home/nkamg/nkrepo/zjp/web_dga/web_file/%s'%(f.filename)
    exe_result = {}
    exe_result =  EXEDetection(file_path)
    query_result=querier.filesha256(f.filename)
    filename=f.filename
    print(query_result)
    if query_result[0] == 0 :
        str_sha256 = query_result[1]
        str_md5 = query_result[2]
        return render_template('malware_file_result.html',status=300 , filename = filename,message="",str_sha256 = str_sha256,str_md5 = str_md5 ,file_sizes =file_sizes,exe_result = exe_result)
    else:
        query_result = query_result[0]
        print(query_result)
        MD5 = query_result[1]
        SHA256 =query_result[2]
        category=query_result[5]
        platform=query_result[6]
        family=query_result[7]
        result=query_result[8]
        filetype=query_result[9]
        return render_template('malware_file_result.html',status=301 ,filename = filename, category = category,platform = platform, family = family, result= result,file_sizes=file_sizes,SHA256=SHA256,MD5=MD5,filetype = filetype,exe_result = exe_result)


@app.route('/malware_result', methods=['POST'])
def detect_url():
    # 1. get url string
    url_str = request.form["url"].strip()
    # 2. validate string
    if url_str == '':
        return render_template("malware_url_result.html",
                           status=400, url=url_str,
                           message="域名不可为空!!")
    validate = re.match(r"^[A-Za-z0-9._\-]*$", url_str)
    if validate == None or '.' not in url_str:
        return render_template("malware_url_result.html",
                               status=401, url=url_str,
                               message="域名格式不正确，域名中只能包含下划线、短横线、点、字母、数字，请输入正确域名！！")


    results = urldetector.multi_predict_single_dname(url_str)
    return render_template("malware_url_result.html", status=200, url=url_str, base_result=results[0],
                           result=results[1])

@app.route('/vsscan',endpoint='vsscan') 
@protected_route
def vs_scan(): 
#获取vscan数据
    vsscan=[]
    conn=pymysql.connect(host = 'localhost',user = 'zjp',passwd = 'Asd147#xYz',db ='kavscan',charset = 'utf8')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM kav')
    vs_data = cursor.fetchall()  
    for row in vs_data:  
      vsscan.append(row) 
    return render_template('vsscan.html', data=vsscan)
 



if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, threaded=True)


