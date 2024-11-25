# -*- coding: utf-8 -*-
import pymysql
import hashlib
import os
import configparser  


config = configparser.ConfigParser()
  
config.read('config.ini') 

host = config.get('mysql', 'host')  
user = config.get('mysql', 'user')  
passwd = config.get('mysql', 'passwd')  
db = config.get('mysql', 'db')  
charset = config.get('mysql', 'charset') 


class Databaseoperation:


    def filesha256(self, name):  
        names = './web_file/%s' % name  
        with open(names, 'rb') as file:  
            str_sha256 = hashlib.sha256(file.read()).hexdigest()  
        with open(names, 'rb') as file:  
            str_md5 = hashlib.md5(file.read()).hexdigest()  
        os.system("rm ./web_file/%s" % (name))  
        data_vs = self.mysqlsha256(str_sha256)  # 注意这里使用 self.mysqlsha256  
        if data_vs == 0 :
            return data_vs, str_sha256, str_md5
        else:
            return data_vs

    def mysqlsha256(self, str_sha256):  
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
                return 0  # 如果没有找到记录，返回0  
            else:  
                return query_result  # 如果找到记录，返回记录数  
  
        finally:  
            conn.close()  # 无论是否发生异常，都确保关闭数据库连接