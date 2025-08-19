#!/usr/bin/env python3
# -*-coding: utf-8-*-

from flask import Flask, request, send_from_directory, jsonify, abort
from configparser import ConfigParser
from dga_detection import MultiModelDetection
from flask_cors import CORS 
import numpy as np 
import json
import os 
import socket
from flask_mysql import Databaseoperation
from file_detect import EXEDetection
from ensemble_predict import run_ensemble_prediction
import pymysql
import subprocess
from api_vt import VTAPI
import logging
import netifaces
from datetime import datetime
import shutil
import time
import hashlib

# -------------------------- 初始化配置 --------------------------
app_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(app_dir, 'log.txt')

# 日志配置
logger = logging.getLogger('my_app')
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
logger.info("应用启动，日志系统初始化成功")

# 读取配置文件
cp = ConfigParser()
cp.read('config.ini')
HOST_IP = cp.get('ini', 'ip')
host = cp.get('mysql', 'host') 
db1 = cp.get('mysql', 'db_category') 
db2 = cp.get('mysql', 'db_family')
db3 = cp.get('mysql', 'db_platform')
user = cp.get('mysql', 'user')  
passwd = cp.get('mysql', 'passwd')   
charset = cp.get('mysql', 'charset')
api_key = cp.get('API','vt_key')
ROW_PER_PAGE = int(cp.get('ini', 'row_per_page'))
detector = MultiModelDetection()

# 初始化数据库和API客户端
querier = Databaseoperation()
VT = VTAPI(api_key)

# Flask应用初始化
app = Flask(__name__) 
CORS(app) 

# -------------------------- 工具函数 --------------------------
def convert_numpy_to_python(obj):  
    if isinstance(obj, dict):  
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}  
    elif isinstance(obj, (list, tuple)):  
        return [convert_numpy_to_python(i) for i in obj]  
    elif isinstance(obj, (np.int64, np.float64)):  
        return obj.item()  
    else:  
        return obj  

def convert_to_serializable(obj):  
    if isinstance(obj, np.ndarray):  
        return obj.tolist()  
    if isinstance(obj, list):  
        return [convert_to_serializable(item) for item in obj]  
    if isinstance(obj, dict):  
        return {key: convert_to_serializable(value) for key, value in obj.items()}  
    return obj  

def check_file_exists(file_path, max_attempts=5, interval=1):
    for i in range(max_attempts):
        if os.path.exists(file_path):
            return True
        time.sleep(interval)
        logger.warning(f"文件 {file_path} 暂不存在，重试 {i+1}/{max_attempts}")
    return False

def safe_delete_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"文件已删除: {file_path}")
        except Exception as e:
            logger.error(f"删除文件失败: {str(e)}")

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"计算SHA256失败: {str(e)}")
        raise

def get_sample_path(sha256):
    if len(sha256) < 5:
        raise ValueError(f"SHA256格式错误: {sha256}")
    sample_dir = os.path.join(
        app_dir, '../samples',
        sha256[0], sha256[1], sha256[2], sha256[3], sha256[4]
    )
    return os.path.join(sample_dir, sha256)

# -------------------------- 请求日志 --------------------------
def log_request():
    try:
        ip = request.headers.get('X-Real-IP', request.remote_addr)
        path = request.path
        method = request.method
        params = request.args.to_dict()
        sensitive_keys = ['password', 'key', 'secret']
        for key in sensitive_keys:
            if key in params:
                params[key] = '***' * 6
        logger.info(f"IP: {ip}, Method: {method}, Path: {path}, Params: {params}")
    except Exception as e:
        logger.error(f"日志记录异常: {str(e)}")

@app.before_request
def before_request_logging():
    log_request()

# -------------------------- 基础路由 --------------------------
@app.route('/test-log', methods=['GET'])
def test_logging():
    logger.info("手动触发日志测试")
    return jsonify({"status": "success", "message": "日志测试完成"})

@app.route('/api/detect_url', methods=['POST'])      
def detect_domain():      
    data = request.json      
    url = data.get('url')      
    if url is None:      
        return jsonify({'error': 'URL is required'}), 400      
      
    result = detector.multi_predict_single_dname(url)    
    if isinstance(result, tuple) and len(result) == 2:    
        result_dict, status_code = result    
        result_dict = convert_numpy_to_python(result_dict)
        return jsonify({'status': '1' if status_code else '0', 'result': result_dict})    
    else:    
        return jsonify({'error': 'Unexpected result format'}), 500  

# -------------------------- 文件上传核心逻辑 --------------------------
UPLOAD_FOLDER = os.path.join(app_dir, '../vue/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"上传目录初始化: {UPLOAD_FOLDER}")

@app.route('/upload', methods=['POST'])  
def upload_file():  
    if 'file' not in request.files:  
        return jsonify({'error': 'No file part'}), 400  
    receive_file = request.files['file']  

    if receive_file.filename == '':  
        return jsonify({'error': 'No selected file'}), 400  

    # 1. 保存上传文件到临时目录
    original_filename = receive_file.filename 
    temp_file_path = os.path.join(UPLOAD_FOLDER, original_filename)  
    try:
        receive_file.save(temp_file_path)
        with open(temp_file_path, 'rb') as f:
            f.read(1)  # 强制同步到磁盘
        logger.info(f"临时文件保存成功: {temp_file_path}")
    except Exception as e:
        logger.error(f"临时文件保存失败: {str(e)}")
        return jsonify({'error': '文件上传失败'}), 500

    if not check_file_exists(temp_file_path):
        logger.error(f"临时文件验证失败: {temp_file_path}")
        return jsonify({'error': '文件上传失败'}), 500

    # 2. 执行文件检测
    try:
        exe_result = run_ensemble_prediction(temp_file_path)
        for key, value in exe_result.items():  
            if isinstance(value, np.ndarray):  
                exe_result[key] = value.tolist() 
        logger.info(f"文件检测完成: {original_filename}")
    except Exception as e:
        logger.error(f"文件检测失败: {str(e)}")
        safe_delete_file(temp_file_path)
        return jsonify({'error': '文件检测失败'}), 500

    # 3. 计算SHA256
    try:
        sha256 = calculate_sha256(temp_file_path)
        logger.info(f"SHA256计算成功: {sha256}")
    except Exception as e:
        logger.error(f"SHA256计算失败")
        safe_delete_file(temp_file_path)
        return jsonify({'error': '文件哈希计算失败'}), 500

    # 4. 移动文件到样本库
    sample_file_path = get_sample_path(sha256)
    sample_dir = os.path.dirname(sample_file_path)
    local_exists = os.path.exists(sample_file_path)

    try:
        os.makedirs(sample_dir, exist_ok=True)
        shutil.move(temp_file_path, sample_file_path)
        logger.info(f"文件移动入库成功: {temp_file_path} -> {sample_file_path}")
        local_exists = True
    except Exception as e:
        logger.error(f"文件移动入库失败: {str(e)}")
        safe_delete_file(temp_file_path)
        return jsonify({'error': '文件入库失败'}), 500

    # 5. 处理结果并返回
    try:
        file_size_bytes = os.path.getsize(sample_file_path)  
        file_size = f"{file_size_bytes / 1024:.2f} KB"  

        # 查询数据库信息
        query_result = querier.mysqlsha256s(sha256) or []
        query_result = convert_to_serializable(query_result)
        query_result_inner = query_result[0] if len(query_result) > 0 else []

        # 解析数据库结果
        query_result_dict = {  
            'MD5': query_result_inner[1] if len(query_result_inner) > 1 else '', 
            'SHA256': sha256,  
            '类型': query_result_inner[5] if len(query_result_inner) > 5 else '', 
            '平台': query_result_inner[6] if len(query_result_inner) > 6 else '', 
            '家族': query_result_inner[7] if len(query_result_inner) > 7 else '' 
        }

        # VT API摘要信息
        VT_API = VT.get_summary(sha256) if api_key else None

        return jsonify({
            'original_filename': original_filename, 
            'query_result': query_result_dict,
            'file_size': file_size,  
            'exe_result': exe_result, 
            'VT_API': VT_API, 
            'local_exists': local_exists
        })
    except Exception as e:
        logger.error(f"结果处理失败: {str(e)}")
        return jsonify({'error': '结果处理失败'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -------------------------- VT API 路由 --------------------------
@app.route('/detection_API/<sha256>')  
def get_detection_API(sha256): 
    sample_file_path = get_sample_path(sha256)
    local_exists = os.path.exists(sample_file_path)
    logger.info(f"VT检测 - 本地样本存在: {local_exists}")

    try:
        if local_exists:
            VT.post_url(sample_file_path)
            json_file_path = VT.get_API_result_detection(sha256, os.path.dirname(sample_file_path))
        else:
            temp_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(sha256[:10])]
            if temp_files:
                upload_file_path = os.path.join(UPLOAD_FOLDER, temp_files[0])
                VT.post_url(upload_file_path)
                json_file_path = VT.get_API_result_detection(sha256, UPLOAD_FOLDER)
            else:
                json_file_path = VT.get_API_result_detection(sha256, UPLOAD_FOLDER)
    except Exception as e:
        logger.error(f"VT API调用失败: {str(e)}")
        return jsonify({'error': 'VT API调用失败'}), 500

    if not json_file_path or not os.path.exists(json_file_path):
        return jsonify({'error': '无法获取检测报告'}), 500

    try:
        with open(json_file_path, 'r') as f:
            scan_result = json.load(f)
        # 更新 detection 列为 1（标记检测报告已入库）
        querier.update_db(sha256, update_type='detection')
    except Exception as e:
        logger.error(f"解析检测报告失败: {str(e)}")
        return jsonify({'error': '解析检测报告失败'}), 500

    results = []
    last_analysis = scan_result['data']['attributes'].get('last_analysis_results', {})
    for engine in last_analysis.values():
        results.append({
            "method": engine.get('method', ''),
            "engine_name": engine.get('engine_name', ''),
            "engine_version": engine.get('engine_version', ''),
            "engine_update": engine.get('engine_update', ''),
            "category": engine.get('category', ''),
            "result": engine.get('result', '')
        })
    return jsonify(results)

@app.route('/behaviour_API/<sha256>')  
def get_behaviour_API(sha256):  
    sample_file_path = get_sample_path(sha256)
    local_exists = os.path.exists(sample_file_path)
    logger.info(f"VT行为分析 - 本地样本存在: {local_exists}")

    try:
        if local_exists:
            VT.post_url(sample_file_path)
            behaviour_file_path = VT.get_API_result_behaviour(sha256, os.path.dirname(sample_file_path))
        else:
            temp_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(sha256[:10])]
            if temp_files:
                upload_file_path = os.path.join(UPLOAD_FOLDER, temp_files[0])
                VT.post_url(upload_file_path)
                behaviour_file_path = VT.get_API_result_behaviour(sha256, UPLOAD_FOLDER)
            else:
                behaviour_file_path = VT.get_API_result_behaviour(sha256, UPLOAD_FOLDER)
    except Exception as e:
        logger.error(f"行为API调用失败: {str(e)}")
        return jsonify({'error': '行为API调用失败'}), 500

    if not behaviour_file_path or not os.path.exists(behaviour_file_path):
        return jsonify({'error': '行为报告不存在'}), 404

    try:
        with open(behaviour_file_path, 'r') as f:
            behaviour_data = json.load(f)
        # 更新 behaviour_summary 列为 1（标记行为报告已入库）
        querier.update_db(sha256, update_type='behaviour')
        return jsonify(behaviour_data.get('data', {}))
    except Exception as e:
        return jsonify({'error': f'解析行为报告失败: {str(e)}'}), 500

# -------------------------- 样本查询与下载 --------------------------
def get_file_path_and_zip(sha256, zip_password="infected"):  
    sample_file_path = get_sample_path(sha256)
    zip_file_path = os.path.join(app_dir, '../data/zips', f'{sha256}.zip')  

    if os.path.exists(sample_file_path):  
        if os.path.exists(zip_file_path):  
            return zip_file_path  
        else:  
            os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)  
            command = [  
                '7z', 'a', '-tzip', f'-p{zip_password}',  
                zip_file_path, sample_file_path  
            ]  
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"压缩文件失败: {e.stderr}")
                return None
            return zip_file_path  
    else:  
        logger.error(f"样本文件不存在: {sample_file_path}")
        return None

# 类别查询
@app.route('/query_category', methods=['POST'])  
def query_virus_category():   
    data = request.json  
    table_name = data.get('tableName', None)  
    if not table_name:  
        return jsonify({'error': '未提供类型名称'}), 400  
    table_name = 'category_' + table_name
    database = db1
    sha256s = querier.mysql(table_name,database)
    if sha256s and sha256s != 0:
        return jsonify({'sha256s': sha256s})
    else:
        return jsonify({'error': '未找到数据'}), 500  

@app.route('/detail_category/<sha256>')  
def get_detail_category(sha256):
    try:
        query_result = querier.mysqlsha256s(sha256) 
        query_result = convert_to_serializable(query_result)
        query_result_inner = query_result[0] if query_result else {}
        query_result_dict = {  
            'MD5': query_result_inner.get(1, ''), 'SHA256': query_result_inner.get(2, ''),  
            '类型': query_result_inner.get(5, ''), '平台': query_result_inner.get(6, ''), 
            '家族': query_result_inner.get(7, ''),
            '文件拓展名': query_result_inner.get(10, ''), '脱壳': query_result_inner.get(11, ''), 
            'SSDEEP': query_result_inner.get(12, '') 
        }
        return jsonify({'query_result': query_result_dict})
    except Exception as e:
        logger.error(f"获取详情失败: {str(e)}")
        return jsonify({'error': '获取详情失败'}), 500

@app.route('/download_category/<sha256>', methods=['GET'])  
def download_file_category(sha256):  
    file_path = get_file_path_and_zip(sha256)  
    if file_path is None:  
        abort(404)
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)

# 家族查询
@app.route('/query_family', methods=['POST'])  
def query_virus_family():  
    data = request.json  
    table_name = data.get('tableName', None)  
    if not table_name:  
        return jsonify({'error': '未提供类型名称'}), 400  
    table_name = 'family_' + table_name  
    database = db2
    sha256s = querier.mysql(table_name,database)
    if sha256s and sha256s != 0:
        return jsonify({'sha256s': sha256s})
    else:
        return jsonify({'error': '未找到数据'}), 500  

@app.route('/detail_family/<sha256>')  
def get_detail_family(sha256):
    try:
        query_result = querier.mysqlsha256s(sha256) 
        query_result = convert_to_serializable(query_result)
        query_result_inner = query_result[0] if query_result else {}
        query_result_dict = {  
            'MD5': query_result_inner.get(1, ''), 'SHA256': query_result_inner.get(2, ''),  
            '类型': query_result_inner.get(5, ''), '平台': query_result_inner.get(6, ''), 
            '家族': query_result_inner.get(7, ''),
            '文件拓展名': query_result_inner.get(10, ''), '脱壳': query_result_inner.get(11, ''), 
            'SSDEEP': query_result_inner.get(12, '') 
        }
        return jsonify({'query_result': query_result_dict})
    except Exception as e:
        logger.error(f"获取详情失败: {str(e)}")
        return jsonify({'error': '获取详情失败'}), 500

@app.route('/download_family/<sha256>', methods=['GET'])  
def download_file_family(sha256):  
    file_path = get_file_path_and_zip(sha256)  
    if file_path is None:  
        abort(404)
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)

# 平台查询
@app.route('/query_platform', methods=['POST'])  
def query_virus_platform():   
    data = request.json  
    table_name = data.get('tableName', None)  
    if not table_name:  
        return jsonify({'error': '未提供类型名称'}), 400  
    table_name = 'platform_' + table_name  
    database = db3
    sha256s = querier.mysql(table_name,database)
    if sha256s and sha256s != 0:
        return jsonify({'sha256s': sha256s})
    else:
        return jsonify({'error': '未找到数据'}), 500

@app.route('/detail_platform/<sha256>')  
def get_detail_platform(sha256):
    try:
        query_result = querier.mysqlsha256s(sha256) 
        query_result = convert_to_serializable(query_result)
        query_result_inner = query_result[0] if query_result else {}
        query_result_dict = {  
            'MD5': query_result_inner.get(1, ''), 'SHA256': query_result_inner.get(2, ''),  
            '类型': query_result_inner.get(5, ''), '平台': query_result_inner.get(6, ''), 
            '家族': query_result_inner.get(7, ''),
            '文件拓展名': query_result_inner.get(10, ''), '脱壳': query_result_inner.get(11, ''), 
            'SSDEEP': query_result_inner.get(12, '') 
        }
        return jsonify({'query_result': query_result_dict})
    except Exception as e:
        logger.error(f"获取详情失败: {str(e)}")
        return jsonify({'error': '获取详情失败'}), 500

@app.route('/download_platform/<sha256>', methods=['GET'])  
def download_file_platform(sha256):  
    file_path = get_file_path_and_zip(sha256)  
    if file_path is None:  
        abort(404)
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)

# SHA256查询
@app.route('/query_sha256', methods=['POST'])  
def query_virus_SHA256(): 
    try:  
        data = request.json  
        sha256 = data.get('tableName', None)  
        if not sha256:  
            return jsonify({'error': '未提供SHA256'}), 400
        query_result = querier.mysqlsha256s(sha256)
        query_result = convert_to_serializable(query_result)
        query_result_inner = query_result[0] if query_result else {}
        query_result_dict = {  
            'MD5': query_result_inner.get(1, ''), 'SHA256': query_result_inner.get(2, ''),  
            '类型': query_result_inner.get(5, ''), '平台': query_result_inner.get(6, ''), 
            '家族': query_result_inner.get(7, ''),
            '文件拓展名': query_result_inner.get(10, ''), '脱壳': query_result_inner.get(11, ''), 
            'SSDEEP': query_result_inner.get(12, '')
        }
        return jsonify({'query_sha256': query_result_dict})
    except pymysql.MySQLError as e:  
        return jsonify({'error': f'数据库错误: {str(e)}'}), 500  
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500  

@app.route('/download_sha256/<sha256>', methods=['GET'])  
def download_file_sha256(sha256):  
    file_path = get_file_path_and_zip(sha256)  
    if file_path is None:  
        abort(404)
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)

# -------------------------- 辅助函数 --------------------------
def get_host_ip():
    try:
        if HOST_IP and HOST_IP not in ['0.0.0.0', '127.0.0.1']:
            return HOST_IP
            
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            if ip not in ['0.0.0.0', '127.0.0.1']:
                return ip
        except:
            pass
            
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info.get('addr')
                    if ip and ip not in ['0.0.0.0', '127.0.0.1']:
                        return ip
        return '127.0.0.1'
    except Exception as e:
        logger.error(f"获取IP地址失败: {str(e)}")
        return HOST_IP or '127.0.0.1'

def update_api_config():
    try:
        actual_ip = get_host_ip()
        port = 5005
        base_url = f"http://{actual_ip}:{port}"
        
        vue_config_path = os.path.join(app_dir, '../vue/public/config.ini')
        os.makedirs(os.path.dirname(vue_config_path), exist_ok=True)
        
        api_cp = ConfigParser()
        api_cp.optionxform = str
        
        if os.path.exists(vue_config_path):
            api_cp.read(vue_config_path)
        
        if not api_cp.has_section('api'):
            api_cp.add_section('api')
        
        api_cp.set('api', 'baseUrl', base_url)
        
        with open(vue_config_path, 'w') as f:
            api_cp.write(f)
        
        logger.info(f"API配置文件已更新: {vue_config_path}, baseUrl: {base_url}")
        return base_url
    except Exception as e:
        logger.error(f"更新API配置文件失败: {str(e)}")
        return None

# -------------------------- 启动应用 --------------------------
if __name__ == '__main__':
    base_url = update_api_config()
    if base_url:
        logger.info(f"服务运行在: {base_url}")
    app.run(host=HOST_IP or '0.0.0.0', port=5005, threaded=True)