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


class Databaseoperation:
    def __init__(self):
        # 初始化数据库连接参数
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.host = config.get('mysql', 'host')
        self.user = config.get('mysql', 'user')
        self.passwd = config.get('mysql', 'passwd')
        self.db = config.get('mysql', 'db')
        self.charset = config.get('mysql', 'charset')

    # 文件检测数据库查询
    def filesha256(self, name):  
        file_path = '../vue/uploads/%s' % name  
        with open(file_path, 'rb') as file:  
            str_sha256 = hashlib.sha256(file.read()).hexdigest()  
        with open(file_path, 'rb') as file:  
            str_md5 = hashlib.md5(file.read()).hexdigest()    
        data_vs = self.mysqlsha256(str_sha256, str_md5)  
        if data_vs == 0:  
            # 构造新目录路径  
            new_dir_path = '../samples/%s/%s/%s/%s/%s' % (  
                str_sha256[0], str_sha256[1], str_sha256[2],  
                str_sha256[3], str_sha256[4] 
            )  
            # 确保目录存在  
            os.makedirs(new_dir_path, exist_ok=True)  
            # 将文件移动到新目录  
            new_file_path = os.path.join(new_dir_path, str_sha256)  
            shutil.move(file_path, new_file_path) 
            # 返回 SHA256 和 MD5 哈希值  
            return str_sha256, str_md5, '0'
        else:  
            # 如果哈希值存在于数据库中，删除原始文件  
            os.remove(file_path) 
            data_vs = data_vs[0] 
            if data_vs[12] == 'nan\r':
                return data_vs, '0'
            else:
                return data_vs, '1'

    def mysqlsha256(self, str_sha256, str_md5):  
        table_prefix = str_sha256[:2]  
        table_prefix = 'sample_' + table_prefix  
        table_name = f"{table_prefix}"  
        print(table_name)  
  
        # 连接数据库  
        conn = pymysql.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db, charset=self.charset)  
  
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
                    return query_result  
        except pymysql.MySQLError as e:  
            print(f"Database error: {e}")  
        finally:  
            conn.close()  

    # 查询sha256列表随机返回20条
    def mysql(self, table_name, database):
        try: 
            print(f"=== mysql方法查询 ===")
            print(f"连接参数: host={self.host}, user={self.user}, database={database}, 表名={table_name}")
        
            # 使用实例变量连接数据库
            conn = pymysql.connect(
                host=self.host,
                user=self.user,
                passwd=self.passwd,
                db=database, 
                charset=self.charset
            )
        
            with conn.cursor() as cursor:
                sql = f"SELECT sha256 FROM `{table_name}` ORDER BY RAND() LIMIT 20"  
                print(f"执行SQL: {sql}")
            
                cursor.execute(sql)
                results = cursor.fetchall()
            
                print(f"查询返回行数: {len(results)}")
                print(f"原始结果: {results[:3]}...")
            
                sha256s = [row[0] for row in results]
                return sha256s
            
        except Exception as e:  
            print(f"查询异常: {str(e)}")
            return 0
        
        finally:  
            if 'conn' in locals() and conn:  
                try:
                    conn.close()
                    print("数据库连接已关闭")
                except Exception as e:
                    print(f"关闭连接异常: {str(e)}")
    
    # API查询成功后更新数据库（支持更新detection或behaviour_summary）
    def update_db(self, str_sha256, update_type):
        """
        更新数据库入库状态
        :param str_sha256: 文件SHA256
        :param update_type: 更新类型，'detection' 或 'behaviour'
        """
        table_prefix = str_sha256[:2]
        table_prefix = 'sample_' + table_prefix 
        table_name = f"{table_prefix}" 
        
        # 根据类型选择要更新的列
        if update_type == 'detection':
            column = 'detection'  # 检测报告入库标记
        elif update_type == 'behaviour':
            column = 'behaviour_summary'  # 行为报告入库标记
        else:
            print(f"无效的更新类型: {update_type}")
            return None

        # 连接数据库（使用实例变量）
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            passwd=self.passwd,
            db=self.db,
            charset=self.charset
        )
        
        try:  
            with conn.cursor() as cursor:  
                # 构建更新SQL语句
                sql = f"UPDATE `{table_name}` SET `{column}` = %s WHERE sha256 = %s;"  
                update_data = ('1', str_sha256)  # 设置为'1'表示已入库
  
                cursor.execute(sql, update_data)  
                conn.commit()  
                print(f"更新成功：{column} 设为 1（SHA256: {str_sha256}）")
                return cursor.rowcount  
  
        except pymysql.Error as e:  
            conn.rollback()  
            print(f"更新数据库失败: {e}")  
            return None  
  
        finally:  
            conn.close()

    def mysqlsha256s(self, str_sha256):  
        # 假设表名格式是固定的，前两个字符作为表名前缀  
        table_prefix = str_sha256[:2]
        table_prefix = 'sample_' + table_prefix 
        table_name = f"{table_prefix}"  

        # 连接数据库  
        conn = pymysql.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db, charset=self.charset)  
  
        try:  
            with conn.cursor() as cursor:  
                # 构建正确的SQL语句  
                sql = "SELECT * FROM `%s` WHERE TRIM(sha256) = %%s;" % table_prefix  
                cursor.execute(sql, (str_sha256,))  
                query_result = cursor.fetchall()  
  
                # 检查查询结果  
                if not query_result: 
                    return 0 
                else:  
                    return query_result  
  
        finally:  
            conn.close()