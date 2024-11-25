#!/usr/bin/env python3
# -*-coding: utf-8 -*-

from flask_sqlalchemy import SQLAlchemy  
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()  

class User(db.Model):  
    id = db.Column(db.Integer, primary_key=True)  
    username = db.Column(db.String(80), unique=True, nullable=False)  
    password_hash = db.Column(db.String(128), nullable=False)  
    is_active = db.Column(db.Boolean, default=True, nullable=False)
#    用于注册时设置密码的哈希值
    def set_password(self, password):  
        self.password_hash = generate_password_hash(password)  
        print("设置")
        print(self.password_hash)
 
    def check_password(self, password):  
        print("check")
        print(self.password_hash,password) 
        return check_password_hash(self.password_hash, password) 

    def get_id(self): 
        return str(self.id)

 