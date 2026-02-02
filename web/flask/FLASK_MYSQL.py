# -*- coding: utf-8 -*-
import pymysql
import hashlib
import os
import configparser
import shutil   


config = configparser.ConfigParser()

config.read('config.ini')

host = config.get('mysql', 'host')
user = config.get('mysql', 'user')
passwd = config.get('mysql', 'passwd')
db = config.get('mysql', 'db')
charset = config.get('mysql', 'charset')

# Read file paths from config
SAMPLE_REPO = config.get('files', 'sample_repo')
UPLOAD_FOLDER_CONFIG = config.get('files', 'upload_folder', fallback='../vue/uploads')

# Resolve upload folder path (relative to this file's directory)
if os.path.isabs(UPLOAD_FOLDER_CONFIG):
    UPLOAD_FOLDER = UPLOAD_FOLDER_CONFIG
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), UPLOAD_FOLDER_CONFIG) 


class Databaseoperation:

    #文件检测数据库查询
    def filesha256(self, name):
        # Use configured upload folder path
        file_path = os.path.join(UPLOAD_FOLDER, name)
        with open(file_path, 'rb') as file:
            str_sha256 = hashlib.sha256(file.read()).hexdigest()
        with open(file_path, 'rb') as file:
            str_md5 = hashlib.md5(file.read()).hexdigest()
        data_vs = self.mysqlsha256(str_sha256,str_md5)# 注意这里使用 self.mysqlsha256
        if data_vs == 0:
            # Build new directory path using configured sample_repo
            prefix = list(str_sha256[:5])
            new_dir_path = os.path.join(SAMPLE_REPO, *prefix)  
            # 确保目录存在  
            os.makedirs(new_dir_path, exist_ok=True)  
            # 将文件移动到新目录  
            new_file_path = os.path.join(new_dir_path, str_sha256)  
            shutil.move(file_path, new_file_path) 
            # 返回 SHA256 和 MD5 哈希值  
            return str_sha256, str_md5,'0'
        else:  
            # 如果哈希值存在于数据库中，删除原始文件  
            os.remove(file_path) 
            data_vs = data_vs[0] 
            if data_vs[12] == 'nan\r':
                return data_vs ,'0'
            else:
                return data_vs ,'1'

    def mysqlsha256(self, str_sha256, str_md5):  
        table_prefix = str_sha256[:2]  
        table_prefix = 'sample_' + table_prefix  
        table_name = f"{table_prefix}"  
        print(table_name)  
  
    # 连接数据库  
        conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, charset=charset)  
  
        try:  
            with conn.cursor() as cursor:  
            # 构建正确的SQL语句  
                sql = f"SELECT * FROM `{table_name}` WHERE TRIM(sha256) = %s;"  
                cursor.execute(sql, (str_sha256,))  
                query_result = cursor.fetchall()    
  
            # 检查查询结果  
                if not query_result:  
                    new_data = ['nan', str_md5, str_sha256, 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan\r','nan','nan']  
                    print(new_data)  
                    sql_insert = f"INSERT INTO `{table_name}` (name, md5, sha256, src_file, date, category, platform, family, result, microsoft_result, filetype, packer, ssdeep, detection , behaviour_summary ,behaviour_mitre_trees) VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s , %s, %s);"  
                    cursor.execute(sql_insert, new_data)  
                    conn.commit()    
                    return 0  
                else:    
                    return query_result  # 如果找到记录，返回记录数  
        except pymysql.MySQLError as e:  
            print(f"Database error: {e}")  
        finally:  
            conn.close()  # 无论是否发生异常，都确保关闭数据库连接
    #查询sha256列表随机返回20条
    def mysql(self, table_name,database):
        try: 
            conn = pymysql.connect(host=host, user=user, passwd=passwd, db=database, charset=charset)
            with conn.cursor() as cursor:
                sql = "SELECT sha256 FROM {} ORDER BY RAND() LIMIT 20".format(table_name)
                cursor.execute(sql)
                results = cursor.fetchall()
                sha256s = [row[0] for row in results]
                return sha256s
        except pymysql.MySQLError as e:  
        # 如果查询失败（例如表不存在），返回错误信息  
            return 0
        finally:  
        # 关闭数据库连接  
            if conn:  
                conn.close()
    
    #API查询成功后 更新数据库
    def update_db(self,str_sha256):
        table_prefix = str_sha256[:2]
        table_prefix ='sample_'+table_prefix 
        table_name = f"{table_prefix}"  # 假设表名后缀是固定的"xx"，请根据实际情况调整 
        conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, charset=charset) 
        try:  
           with conn.cursor() as cursor:  
            # 构建更新SQL语句  
                sql = f"UPDATE `{table_name}` SET vt = %s WHERE sha256 = %s;"  
            # 准备要更新的数据  
                update_data = ('1', str_sha256)  
  
            # 执行更新操作  
                cursor.execute(sql, update_data)  
  
            # 提交事务以确保更改被保存  
                conn.commit()  
  
            # 可选：返回受影响的行数（即更新的行数）  
                return cursor.rowcount  
  
        except pymysql.Error as e:  
          # 如果发生错误，回滚事务并打印错误信息  
                conn.rollback()  
                print(f"An error occurred: {e}")  
                return None  # 或者根据需要返回其他错误指示  
  
        finally:  
         # 无论是否发生异常，都关闭数据库连接  
                conn.close()

    def mysqlsha256s(self, str_sha256):  
        # 假设表名格式是固定的，前两个字符作为表名前缀  
        table_prefix = str_sha256[:2]
        table_prefix ='sample_'+table_prefix 
        table_name = f"{table_prefix}"  # 假设表名后缀是固定的"xx"，请根据实际情况调整 

  
    # 连接数据库  
        conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db, charset=charset)  
  
        try:  
            with conn.cursor() as cursor:  
            # 构建正确的SQL语句  
                sql = "SELECT * FROM `%s` WHERE TRIM(sha256) = %%s;" % table_prefix # 注意这里的%%s，因为%是字符串格式化的特殊字符  
                cursor.execute(sql, (str_sha256,))  # 使用参数化查询  
                query_result = cursor.fetchall()  
  
        # 检查查询结果  
            if not query_result: 
                return 0 
            else:  
                return query_result  # 如果找到记录，返回记录数  
  
        finally:  
            conn.close()  # 无论是否发生异常，都确保关闭数据库连接
    