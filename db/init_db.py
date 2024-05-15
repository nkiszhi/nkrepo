#!/usr/bin/env python3
# -*-coding: utf-8 -*-
# Initialize the database and create tables store malicious domain name samples and web_users.

import mysql.connector  
import argparse  
from datetime import datetime,timedelta

database_name="nkrepo"
cnx = None 

#创建数据库nkrepo
def create_database():  
    try:  
        cursor = cnx.cursor()  
        # 创建数据库（如果尚未存在）  
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name};")  
        cnx.commit()  
        print(f"[o] Database {database_name} connected successfully.")  
    except mysql.connector.Error as err:  
        print(f"[!] Failed to create database: {err}")  

def create_table_sample():  
    try:  
        cursor = cnx.cursor()  
        # 选择数据库  
        cursor.execute(f"USE {database_name};")  
        create_table_template = """  
        CREATE TABLE IF NOT EXISTS `{table_name}` (  
            id INT AUTO_INCREMENT PRIMARY KEY,  
            name VARCHAR(255), 
            md5 VARCHAR(255),  
            sha256 VARCHAR(255), 
            src_file VARCHAR(255), 
            date VARCHAR(255), 
            category VARCHAR(255), 
            platform VARCHAR(255),
            family VARCHAR(255),
            result VARCHAR(255),
            filetype VARCHAR(255),
            packer VARCHAR(255),
            ssdeep VARCHAR(255)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;  
        """  
        # 遍历00到FF并创建表  
        for i in range(256):  
            sample_name = format(i, '02x')  # 将数字转换为两位十六进制字符串，如'00'到'ff'  
            table_name = f'sample_{sample_name}'  # 表名前缀为'sample_'  
            create_table_sql = create_table_template.format(table_name=table_name)  
            
            cursor.execute(f"SHOW TABLES LIKE '{table_name}';")  
            if cursor.fetchone() is not None:  
                print(f"[i] Table {table_name} already exists.")  
            else:  
        # 执行SQL语句创建表  
                create_table_sql = create_table_template.format(table_name=table_name)  
                try:  
                    cursor.execute(create_table_sql)  
                    print(f"[o] Table {table_name} created successfully.")  
                except Error as err:  
                    print(f"[!] Failed to create table {table_name}: {err}")            
        # 提交更改  
        cnx.commit()  
  
    except mysql.connector.Error as err:  
        print(f"[!] Failed to  connect to MySQL: {err}")  
 
    print("[o] All sample_xy tables created successfully!")  

#创建数据表domain
def create_table_domain():  
    table_name=datetime.now().date().strftime('domain_%Y%m%d')
    try:  
        cursor = cnx.cursor()  
        cursor.execute(f"USE {database_name};")  
        create_table_template = """  
        CREATE TABLE IF NOT EXISTS `{table_name}` (  
            id INT AUTO_INCREMENT PRIMARY KEY,  
            name VARCHAR(255), 
            category VARCHAR(255), 
            source VARCHAR(255)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;  
        """  
        create_table_sql = create_table_template.format(table_name=table_name)
        cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
        if cursor.fetchone() is None:  
            # 执行SQL语句创建表  
            try:  
                cursor.execute(create_table_sql)  
                print(f"[o] Table {table_name} created successfully.")  
            except Error as err:  
                print(f"[!] Failed to create table {table_name}: {err}")  
        else:  
            print(f"[i] Table {table_name} already exists.")  
        cnx.commit()  
  
    except mysql.connector.Error as err:  
        print(f"[!] Failed to  connect to MySQL: {err}")  

#创建数据表user
def create_table_user():  
    table_name="user"
    try:  
        cursor = cnx.cursor()  
        # 选择数据库  
        cursor.execute(f"USE {database_name};")  
        # 创建表的SQL模板  
        create_table_template = """  
        CREATE TABLE IF NOT EXISTS `{table_name}` (  
            id INT AUTO_INCREMENT PRIMARY KEY,  
            username VARCHAR(80) NOT NULL UNIQUE,  
            password_hash VARCHAR(128) NOT NULL,  
            is_active TINYINT(1) NOT NULL DEFAULT 1 
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;  
        """  
        cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
        create_table_sql = create_table_template.format(table_name=table_name)
        if cursor.fetchone() is None:  
            # 执行SQL语句创建表  
            try:  
                cursor.execute(create_table_sql)  
                print(f"[o] Table {table_name} created successfully.")  
            except Error as err:  
                    print(f"[!] Failed to create table {table_name}: {err}")  
        else:  
            print(f"[i] Table {table_name} already exists.")   
            cnx.commit()  
  
    except mysql.connector.Error as err:  
        print(f"[!] Failed to  connect to MySQL: {err}")  
def main():  
    parser = argparse.ArgumentParser(description='Initialize the database and create tables store malicious domain name samples and web_users.',add_help=False)  
    parser.add_argument('-u', '--user', required=True, help='MySQL username')  
    parser.add_argument('-p', '--password', required=True, help='MySQL password')  
    parser.add_argument('-h', '--host', required=True, help='MySQL host')  
    
    global cnx
    try:  
        args = parser.parse_args()  
    except argparse.ArgumentError:  
        parser.print_help()  
    
    db_config = {  
        'user': args.user,  
        'password': args.password,  
        'host': args.host,  
    }  
    print(db_config)

    try:  
        cnx = mysql.connector.connect(**db_config)  
        create_database() 
        create_table_sample()
        create_table_domain()
        create_table_user()
 
    except mysql.connector.Error as err:  
        print(f"[!] Failed to connect MySQL: {err}")  
        
    finally:  
        if 'cnx' in locals() and cnx.is_connected():
            cnx.close()  
            print("[o] MySQL connection is closed.")
  
if __name__ == "__main__":  
    main()
