# -*- coding: utf-8 -*-
import pymysql
import hashlib
import os
import configparser
import shutil
import re
from datetime import datetime

MYSQL_IDENTIFIER_RE = re.compile(r"\A[A-Za-z0-9_]{1,64}\Z")


def _split_config_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def _sanitize_mysql_identifier(identifier):
    if not MYSQL_IDENTIFIER_RE.fullmatch(identifier or ""):
        raise ValueError(f"非法MySQL标识符: {identifier}")
    return identifier


def _infer_file_kind(db_name):
    lowered = (db_name or "").lower()
    for kind in ("elf", "pe", "others"):
        if lowered.endswith(f"_{kind}") or f"_{kind}_" in lowered:
            return kind
    return ""


def _infer_sample_class(db_name):
    lowered = (db_name or "").lower()
    if lowered.startswith("malicious"):
        return "malicious"
    if lowered.startswith("benign"):
        return "benign"
    return ""

class Databaseoperation:
    def __init__(self):
        """初始化数据库配置"""
        # 读取配置文件
        config = configparser.ConfigParser()
        if not os.path.exists('config.ini'):
            raise FileNotFoundError("配置文件 config.ini 不存在，请检查路径")
        config.read('config.ini', encoding='utf-8')

        # 读取数据库配置并设置默认值
        self.host = config.get('mysql', 'host', fallback='localhost')
        self.user = config.get('mysql', 'user', fallback='root')
        self.passwd = config.get('mysql', 'passwd', fallback='')
        self.port = config.getint('mysql', 'port', fallback=3306)
        self.db = config.get('mysql', 'db', fallback='malicious')
        self.malicious_dbs = _split_config_list(config.get('mysql', 'malicious_dbs', fallback=''))
        self.benign_dbs = _split_config_list(config.get('mysql', 'benign_dbs', fallback=''))
        if not self.malicious_dbs:
            self.malicious_dbs = [self.db]
        if not self.benign_dbs:
            legacy_benign = config.get('mysql', 'db_benign', fallback='')
            self.benign_dbs = [legacy_benign] if legacy_benign else []
        self.sample_dbs = self._build_sample_db_infos()
        self.db_web = config.get('mysql', 'db_web', fallback='webdatadb')
        self.charset = config.get('mysql', 'charset', fallback='utf8mb4')

        # 读取文件路径配置
        self.upload_dir = config.get('paths', 'upload_dir', fallback='uploads')
        self.web_upload_dir = config.get('paths', 'web_upload_dir', fallback='../../../data/web_upload_file')

    def _build_sample_db_infos(self):
        infos = []
        for db_name in self.malicious_dbs:
            infos.append({
                'db_name': db_name,
                'sample_class': 'malicious',
                'file_kind': _infer_file_kind(db_name)
            })
        for db_name in self.benign_dbs:
            infos.append({
                'db_name': db_name,
                'sample_class': 'benign',
                'file_kind': _infer_file_kind(db_name)
            })
        return infos

    def _connect(self, database):
        return pymysql.connect(
            host=self.host,
            user=self.user,
            passwd=self.passwd,
            port=self.port,
            db=database,
            charset=self.charset,
            cursorclass=pymysql.cursors.DictCursor
        )

    def _decorate_rows(self, rows, db_name, sample_class=None, file_kind=None, is_upload=False):
        decorated = []
        for row in rows:
            if isinstance(row, dict):
                item = dict(row)
                item['__source_db'] = db_name
                item['__sample_class'] = sample_class or _infer_sample_class(db_name)
                item['__file_kind'] = file_kind or _infer_file_kind(db_name)
                item['__is_upload'] = is_upload
                decorated.append(item)
            else:
                decorated.append(row)
        return decorated

    def _query_sha256_in_db(self, database, table_name, sha256, sample_class=None, file_kind=None, is_upload=False):
        _sanitize_mysql_identifier(database)
        _sanitize_mysql_identifier(table_name)

        conn = None
        try:
            conn = self._connect(database)
            with conn.cursor() as cursor:
                sql = f"SELECT * FROM `{table_name}` WHERE TRIM(sha256) = %s;"
                cursor.execute(sql, (sha256,))
                rows = cursor.fetchall()
                if rows:
                    return self._decorate_rows(rows, database, sample_class, file_kind, is_upload)
        except pymysql.MySQLError as e:
            print(f"数据库查询错误: db={database}, table={table_name}, error={e}")
        finally:
            if conn:
                conn.close()
        return []

    def filesha256(self, name):
        """
        计算文件SHA256/MD5，查询数据库并处理文件
        :param name: 上传的文件名
        :return: 分支1 (str_sha256, str_md5, '0') | 分支2 (data_vs, '0'/'1')
        """
        # 使用配置文件中的上传目录
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', self.upload_dir, name))
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误：文件 {file_path} 不存在")
            return None, None, '0'
        
        # 读取文件并计算哈希（仅读取一次文件）
        try:
            with open(file_path, 'rb') as file:  
                file_content = file.read()
                str_sha256 = hashlib.sha256(file_content).hexdigest()  
                str_md5 = hashlib.md5(file_content).hexdigest()    
        except Exception as e:
            print(f"读取文件失败：{e}")
            return None, None, '0'
        
        # 查询数据库
        data_vs = self.mysqlsha256(str_sha256, str_md5, name)
        
        # 分支1：数据库无记录 → 移动文件并返回基础信息
        if data_vs == 0:
            # 使用配置文件中的web上传目录
            new_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', self.web_upload_dir))
            os.makedirs(new_dir_path, exist_ok=True)  
            new_file_path = os.path.join(new_dir_path, str_sha256)  
            
            try:
                shutil.move(file_path, new_file_path) 
            except Exception as e:
                print(f"移动文件失败：{e}")
            
            return str_sha256, str_md5, '0'
        
        # 分支2：数据库有记录 → 删除临时文件并判断has_vt
        else:  
            try:
                os.remove(file_path) 
            except Exception as e:
                print(f"删除文件失败：{e}")
            
            data_vs = data_vs[0] 
            
            # 核心修复：兼容字典/元组游标，安全获取has_vt字段
            try:
                # 优先用字段名（字典游标）
                has_vt = data_vs['has_vt']
            except (KeyError, TypeError):
                # 兼容元组游标（表结构中has_vt是第19列，下标18）
                has_vt = data_vs[18] if len(data_vs) >= 19 else None
            
            # 处理has_vt值（兼容int/None/空字符串）
            has_vt_str = str(has_vt).strip() if has_vt is not None else ""
            if not has_vt_str or has_vt_str == '0':            
                return data_vs, '0'            
            else:
                return data_vs, '1'

    def mysqlsha256(self, str_sha256, str_md5, file_name):
        """
        查询SHA256是否存在于数据库，不存在则插入到web库
        :param str_sha256: 文件SHA256值
        :param str_md5: 文件MD5值
        :param file_name: 原始文件名
        :return: 查询结果 | 0（插入成功）
        """
        # 生成表名（sample_前缀+sha256前两位）
        table_prefix = str_sha256[:2]  
        table_name = f"sample_{table_prefix}"  

        # 1. 先查询六个正式样本库
        for db_info in self.sample_dbs:
            query_result = self._query_sha256_in_db(
                db_info['db_name'],
                table_name,
                str_sha256,
                db_info['sample_class'],
                db_info['file_kind'],
                False
            )
            if query_result:
                return query_result
        
        # 2. 再查询web上传库（webdatadb）
        conn_web = None
        try:
            conn_web = self._connect(self.db_web)
            with conn_web.cursor() as cursor_web:
                # 查询是否存在
                sql_web = f"SELECT * FROM `{table_name}` WHERE TRIM(sha256) = %s;"
                cursor_web.execute(sql_web, (str_sha256,))
                query_result_web = cursor_web.fetchall()
                
                # 存在则返回结果
                if query_result_web:
                    return self._decorate_rows(query_result_web, self.db_web, 'upload', 'upload', True)
                
                # 不存在则插入新记录（适配表结构的9个字段）
                current_date = datetime.now().strftime('%Y-%m-%d')
                new_data = [
                    file_name,          # name (varchar(100))
                    0,                  # length (int，默认0)
                    str_sha256,         # sha256 (char(64))
                    str_md5,            # md5 (char(32))
                    '',                 # ssdeep (varchar(150))
                    '',                 # vhash (varchar(120))
                    '',                 # authentihash (char(64))
                    '',                 # imphash (char(32))
                    '',                 # rich_header_hash (varchar(64))
                    'web',              # source (varchar(20))
                    current_date,       # date (date)
                    '',                 # category (varchar(255))
                    '',                 # platform (varchar(255))
                    '',                 # family (varchar(255))
                    '',                 # kav_result (varchar(255))
                    '',                 # defender_result (varchar(255))
                    '',                 # filetype (varchar(255))
                    '',                 # packer (varchar(255))
                    0,                  # has_vt (tinyint(1))
                    0,                  # has_vt_summary (tinyint(1))
                    0                   # has_vt_mitre (tinyint(1))
                ]  
                
                # 插入语句（严格匹配表结构字段）
                sql_insert = f"""
                INSERT INTO `{table_name}` (
                    name, length, sha256, md5, ssdeep, vhash, authentihash, imphash, rich_header_hash,
                    source, date, category, platform, family, kav_result, defender_result,
                    filetype, packer, has_vt, has_vt_summary, has_vt_mitre
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """  
                cursor_web.execute(sql_insert, new_data)  
                conn_web.commit()    
                return 0  # 插入成功返回0
        except pymysql.MySQLError as e:  
            print(f"Web库操作错误: {e}")  
            return 0
        finally:  
            if conn_web:
                conn_web.close()

    def mysql(self, table_name, database):
        """
        随机返回表中20条SHA256值
        :param table_name: 表名
        :param database: 数据库名
        :return: SHA256列表 | 空列表
        """
        conn = None
        try: 
            conn = self._connect(database)
            with conn.cursor() as cursor:
                # 表名安全校验（防止SQL注入）
                if not table_name.startswith('sample_'):
                    raise ValueError("非法的表名前缀，仅允许sample_开头的表")
                
                sql = f"SELECT sha256 FROM `{table_name}` ORDER BY RAND() LIMIT 20"
                cursor.execute(sql)
                results = cursor.fetchall()
                sha256s = [row.get('sha256') if isinstance(row, dict) else row[0] for row in results]
                return sha256s
        except pymysql.MySQLError as e:  
            print(f"随机查询错误: {e}")
            return []
        finally:  
            if conn:  
                conn.close()
    
    def update_db(self, str_sha256):
        """
        更新数据库中has_vt相关字段
        :param str_sha256: 文件SHA256值
        :return: 受影响行数 | 0
        """
        table_prefix = str_sha256[:2]
        table_name = f"sample_{table_prefix}"  
        
        # 1. 先更新六个正式样本库
        for db_info in self.sample_dbs:
            conn = None
            try:
                conn = self._connect(db_info['db_name'])
                with conn.cursor() as cursor:
                    try:
                        sql = f"UPDATE `{table_name}` SET has_vt = %s, has_vt_summary = %s WHERE sha256 = %s;"
                        update_data = (1, 1, str_sha256)
                        cursor.execute(sql, update_data)
                    except pymysql.Error:
                        conn.rollback()
                        sql = f"UPDATE `{table_name}` SET has_vt = %s WHERE sha256 = %s;"
                        update_data = (1, str_sha256)
                        cursor.execute(sql, update_data)
                    conn.commit()
                    affected = cursor.rowcount
                    if affected > 0:
                        return affected
            except pymysql.Error as e:
                print(f"样本库更新错误: db={db_info['db_name']}, error={e}")
            finally:
                if conn:
                    conn.close()
        
        # 2. 再更新web上传库
        conn_web = None
        try:
            conn_web = self._connect(self.db_web)
            with conn_web.cursor() as cursor:
                # 修正字段名：detection -> has_vt, behaviour_summary -> has_vt_summary
                sql = f"UPDATE `{table_name}` SET has_vt = %s, has_vt_summary = %s WHERE sha256 = %s;"  
                update_data = (1, 1, str_sha256)  
                cursor.execute(sql, update_data)  
                conn_web.commit()  
                return cursor.rowcount  
        except pymysql.Error as e:  
            print(f"Web库更新错误: {e}")  
            return 0
        finally:  
            if conn_web:  
                conn_web.close()

    def mysqlsha256s(self, str_sha256):  
        """
        单独查询指定SHA256的记录
        :param str_sha256: 文件SHA256值
        :return: 查询结果 | 0
        """
        table_prefix = str_sha256[:2]
        table_name = f"sample_{table_prefix}"  

        # 1. 先查六个正式样本库
        for db_info in self.sample_dbs:
            query_result = self._query_sha256_in_db(
                db_info['db_name'],
                table_name,
                str_sha256,
                db_info['sample_class'],
                db_info['file_kind'],
                False
            )
            if query_result:
                return query_result
        
        # 2. 再查web上传库
        conn_web = None
        try:
            conn_web = self._connect(self.db_web)
            with conn_web.cursor() as cursor:  
                sql = f"SELECT * FROM `{table_name}` WHERE TRIM(sha256) = %s;"
                cursor.execute(sql, (str_sha256,))  
                query_result = cursor.fetchall()  
                if query_result:  
                    return self._decorate_rows(query_result, self.db_web, 'upload', 'upload', True)
                else:  
                    return 0 
        except pymysql.MySQLError as e:
            print(f"Web库查询错误: {e}")
            return 0
        finally:  
            if conn_web:  
                conn_web.close()

# 测试代码（可选，运行时注释掉）
if __name__ == "__main__":
    try:
        # 初始化数据库操作类
        db_op = Databaseoperation()
        print("数据库配置加载成功 ✅")
        
        # 测试文件哈希计算（替换为实际文件名）
        # result = db_op.filesha256("test_file.exe")
        # print(f"测试结果: {result}")
    except Exception as e:
        print(f"初始化失败 ❌: {e}")
