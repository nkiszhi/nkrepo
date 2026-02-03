from flask import Flask, request, send_from_directory, url_for, jsonify, abort
from flask_cors import CORS
import numpy as np
import json
import os
import pymysql
import subprocess

# Import configuration from config.py
from config import (
    Config,
    HOST_IP, PORT, ROW_PER_PAGE,
    SAMPLE_REPO, ZIP_STORAGE, UPLOAD_FOLDER,
    DB_CATEGORY as db1, DB_FAMILY as db2, DB_PLATFORM as db3,
    VT_API_KEY as api_key
)

from dga_detection import MultiModelDetection
from db_operations import DatabaseOperation
from file_detect import EXEDetection
from ensemble_predict import run_ensemble_prediction
from api_vt import VirusTotalAPI

db_ops = DatabaseOperation()
vt_api = VirusTotalAPI()
dga_detector = MultiModelDetection()
 


  
app = Flask(__name__) 
CORS(app) 

#url检测
  
def convert_numpy_to_python(obj):  
    if isinstance(obj, dict):  
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}  
    elif isinstance(obj, (list, tuple)):  
        return [convert_numpy_to_python(i) for i in obj]  
    elif isinstance(obj, (np.int64, np.float64)):  # 假设导入了numpy为np  
        return obj.item()  # 对于NumPy整数和浮点数，使用.item()转换为Python类型  
    else:  
        return obj  
  
@app.route('/api/detect', methods=['POST'])      
def detect_domain():      
    data = request.json      
    url = data.get('url')      
    if url is None:      
        return jsonify({'error': 'URL is required'}), 400      
      
    result = dga_detector.multi_predict_single_dname(url)    
    if isinstance(result, tuple) and len(result) == 2:    
        # 假设 result 是一个包含字典和整数的元组  
        result_dict, status_code = result    
        # 递归地将字典中的所有NumPy类型转换为Python原生类型  
        result_dict = convert_numpy_to_python(result_dict)
        print(result_dict)  
  
        # 现在可以安全地序列化 result_dict 和 status_code  
        try:    
            return jsonify({'status': '1' if status_code else '0', 'result': result_dict})    
        except TypeError as e:    
            # 如果仍然发生错误，打印更详细的错误信息以进行调试  
            print(f"Error during serialization: {e}")    
            # 这里可以添加更详细的调试代码或返回一个错误响应  
            return jsonify({'error': 'Failed to serialize result'}), 500    
    else:    
        # 如果 result 不是预期的格式，返回一个错误响应  
        return jsonify({'error': 'Unexpected result format'}), 500  
#==========================================================================================================
#文件检测
# Upload folder is configured in config.py
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def convert_to_serializable(obj):  
    if isinstance(obj, np.ndarray):  
        return obj.tolist()  
    if isinstance(obj, list):  
        return [convert_to_serializable(item) for item in obj]  
    if isinstance(obj, dict):  
        return {key: convert_to_serializable(value) for key, value in obj.items()}  
    return obj 


# 确保上传文件夹存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/upload', methods=['POST'])  
def upload_file():  
    if 'file' not in request.files:  
        return jsonify({'error': 'No file part'}), 400  
    receive_file = request.files['file']  
  
    if receive_file.filename == '':  
        return jsonify({'error': 'No selected file'}), 400  
  
    original_filename = receive_file.filename 
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)  
    receive_file.save(file_path) 
    file_size_bytes = os.path.getsize(file_path)  
    file_size_kb = file_size_bytes / 1024  
    file_size = format(file_size_kb, '.2f') + " KB"  
    exe_result = run_ensemble_prediction(file_path)
    for key, value in exe_result.items():  
        if isinstance(value, np.ndarray):  
            exe_result[key] = value.tolist() 
    query_result = db_ops.filesha256(original_filename)
    if len(query_result) == 2:
        query_result = convert_to_serializable(query_result) 
        query_result_inner = query_result[0]
        query_result_dict = {  
            'MD5': query_result_inner[1],  
            'SHA256': query_result_inner[2],  
            '类型': query_result_inner[5],  
            '平台': query_result_inner[6],  
            '家族': query_result_inner[7] 
        }
        VT_API = query_result[1]
        return jsonify({
            'original_filename': original_filename,  
            'query_result': query_result_dict, 
            'file_size': file_size,  
            'exe_result': exe_result,
            'VT_API': VT_API
        }) 
    else:
         query_result = convert_to_serializable(query_result)
         query_result_dict = {  
            'MD5': query_result[1],  
            'SHA256': query_result[0]  
        }
         VT_API = query_result[2]
         
         return jsonify({ 'original_filename': original_filename, 'query_result': query_result_dict, 'file_size': file_size,  'exe_result': exe_result,'VT_API': VT_API}) 
    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
#==========================================================================================================
#API查询detection
@app.route('/detection_API/<sha256>')
def get_detection_API(sha256):
    # Build sample directory path using Config utility
    sample_dir_path = str(Config.get_sample_dir(sha256))
    json_file_path = vt_api.get_API_result_detection(sha256, api_key, sample_dir_path)
    print(json_file_path)
    if json_file_path == 500:
       return 500
    else:
      up_mysql=db_ops.update_db(sha256)
#      print(f"成功更新，影响了 {up_mysql} 行")
      with open(json_file_path, 'r') as file:  
          scan_result = json.load(file)  
       # 验证数据结构  
      if 'data' not in scan_result or 'attributes' not in scan_result['data']:
          return jsonify({'error': 'Missing required data in JSON file'}), 400  

      results = []  
      if 'last_analysis_results' not in scan_result['data']['attributes']:
          last_analysis_results = scan_result['data']['attributes']['results']
      else:
          last_analysis_results = scan_result['data']['attributes']['last_analysis_results']  
      for engine in last_analysis_results.values():  
          results.append({  
              "method": engine['method'],
              "engine_name": engine['engine_name'],
              "engine_version": engine['engine_version'],
              "engine_update": engine['engine_update'],
              "category": engine['category'],
              "result": engine['result']               
          })  
      return jsonify(results)
#==============================================================================================================================
#API查询  behaviour
@app.route('/behaviour_API/<sha256>')
def get_behaviour_API(sha256):
    # Build sample directory path using Config utility
    sample_dir_path = str(Config.get_sample_dir(sha256))
    behaviour_file_path = vt_api.get_API_result_behaviour(sha256, api_key, sample_dir_path)  # 确保 vt_api.get_API_result_behaviour 正确处理并返回文件路径或错误信息 
    print(behaviour_file_path)
    try:  
        with open(behaviour_file_path, 'r') as file:  
            behaviour_scan = json.load(file)  
            behaviour_data = behaviour_scan['data'] 
            print(behaviour_data) 
            return jsonify(behaviour_data)  
    except FileNotFoundError:  
        abort(404, description="Behaviour file not found")  
    except json.JSONDecodeError:  
        abort(400, description="Error decoding JSON file")  
    except Exception as e:  
        abort(500, description=f"Internal Server Error: {str(e)}") 


     
      
#==============================================================================================================================
def get_file_path_and_zip(sha256, zip_password="infected"):
    # Build file path using Config utility
    file_path = str(Config.get_sample_path(sha256))
    # Build ZIP file path using Config utility
    zip_file_path = str(Config.get_zip_path(sha256))  
  
    # 检查原始文件是否存在  
    if os.path.exists(file_path):  
        # 检查ZIP文件是否已存在  
        if os.path.exists(zip_file_path):  
            # 如果ZIP文件已存在，直接返回ZIP文件路径  
            return zip_file_path  
        else:  
            # 如果ZIP文件不存在，则使用7z命令创建它  
            # 确保ZIP文件所在的目录存在  
            os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)  
  
            # Use 7z command to create encrypted ZIP file
            # Check for 7z or 7za (Linux alternative)
            zip_cmd = None
            for cmd in ['7z', '7za']:
                try:
                    subprocess.run([cmd, '--help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    zip_cmd = cmd
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

            if zip_cmd is None:
                raise FileNotFoundError("7zip is not installed. On Ubuntu: sudo apt install p7zip-full")

            command = [
                zip_cmd, 'a', '-tzip', '-p{}'.format(zip_password),
                zip_file_path, file_path
            ]
            subprocess.run(command, check=True)  
  
            # 返回新创建的ZIP文件路径  
            return zip_file_path  
    else:  
        # 如果原始文件不存在，则返回None  
        return None


#===============================================================================================================================
# category搜索

@app.route('/query_category', methods=['POST'])  
def query_virus_category():   
        # 从请求中获取表名
        results = []  
        data = request.json  
        table_name = data.get('tableName', None)  
        if not table_name:  
            return jsonify({'error': '未提供类型名称'}), 400  
        # 构造完整的表名  
        table_name = 'category_' + table_name
        database = db1
        sha256s = db_ops.mysql(table_name,database)
        if not sha256s == 0:
            return jsonify({'sha256s': sha256s})
        else:
            return jsonify({'error': 0}), 500  
           
@app.route('/detail_category/<sha256>')  
def get_detail_category(sha256):
    print(sha256)
    query_result = db_ops.mysqlsha256s(sha256) 
    query_result = convert_to_serializable(query_result)
    query_result_inner = query_result[0]
    query_result_dict = {  'MD5': query_result_inner[1],  'SHA256': query_result_inner[2],  '类型': query_result_inner[5],  '平台': query_result_inner[6],  '家族': query_result_inner[7],'文件拓展名':query_result_inner[10] , '脱壳':query_result_inner[11],'SSDEEP':query_result_inner[12] }
    print(query_result_dict)
    return jsonify({'query_result': query_result_dict})


@app.route('/download_category/<sha256>', methods=['GET'])  
def download_file_category(sha256):  
    file_path = get_file_path_and_zip(sha256)  
    if file_path is None:  
        abort(404)
  
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)

#===============================================================================================================================
# family搜索


@app.route('/query_family', methods=['POST'])  
def query_virus_family():  
        # 从请求中获取表名
        results = []  
        data = request.json  
        table_name = data.get('tableName', None)  
        if not table_name:  
            return jsonify({'error': '未提供类型名称'}), 400  
  
        # 构造完整的表名  
        table_name = 'family_' + table_name  
        database = db2
        sha256s = db_ops.mysql(table_name,database)
        if not sha256s == 0:
            return jsonify({'sha256s': sha256s})
        else:
            return jsonify({'error': 0}), 500  
            conn.close()  
  
@app.route('/detail_family/<sha256>')  
def get_detail_family(sha256):
    query_result = db_ops.mysqlsha256s(sha256) 
    query_result = convert_to_serializable(query_result)
    print(query_result)
    print('111111111111111111')
    query_result_inner = query_result[0]
    query_result_dict = {  'MD5': query_result_inner[1],  'SHA256': query_result_inner[2],  '类型': query_result_inner[5],  '平台': query_result_inner[6],  '家族': query_result_inner[7],'文件拓展名':query_result_inner[10] , '脱壳':query_result_inner[11],'SSDEEP':query_result_inner[12] }
    print(query_result_dict)
    return jsonify({'query_result': query_result_dict})


@app.route('/download_family/<sha256>', methods=['GET'])  
def download_file_family(sha256):  
    file_path = get_file_path_and_zip(sha256)  
    if file_path is None:  
        abort(404)
  
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)

#===============================================================================================================================
# platform搜索

@app.route('/query_platform', methods=['POST'])  
def query_virus_platform():   
        # 从请求中获取表名
        results = []  
        data = request.json  
        table_name = data.get('tableName', None)  
        if not table_name:  
            return jsonify({'error': '未提供类型名称'}), 400  
  
        # 构造完整的表名  
        table_name = 'platform_' + table_name  
        database = db3
        sha256s = db_ops.mysql(table_name,database)
        if not sha256s == 0:
            return jsonify({'sha256s': sha256s})
        else:
            return jsonify({'error': 0}), 500
              
@app.route('/detail_platform/<sha256>')  
def get_detail_platform(sha256):
    query_result = db_ops.mysqlsha256s(sha256) 
    query_result = convert_to_serializable(query_result)
    query_result_inner = query_result[0]
    query_result_dict = {  'MD5': query_result_inner[1],  'SHA256': query_result_inner[2],  '类型': query_result_inner[5],  '平台': query_result_inner[6],  '家族': query_result_inner[7],'文件拓展名':query_result_inner[10] , '脱壳':query_result_inner[11],'SSDEEP':query_result_inner[12] }
    print(query_result_dict)
    return jsonify({'query_result': query_result_dict})


@app.route('/download_platform/<sha256>', methods=['GET'])  
def download_file_platform(sha256):  
    file_path = get_file_path_and_zip(sha256)  
    if file_path is None:  
        abort(404)
  
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)

#===============================================================================================================================
# SHA256搜索

@app.route('/query_sha256', methods=['POST'])  
def query_virus_SHA256(): 
    try:  
        # 从请求中获取表名
        results = []  
        data = request.json  
        sha256 = data.get('tableName', None)  
        if not sha256:  
            return jsonify({'error': '未提供类型名称'}), 400
        query_result = db_ops.mysqlsha256s(sha256)
        query_result = convert_to_serializable(query_result)
        query_result_inner = query_result[0]
        query_result_dict = {  'MD5': query_result_inner[1],  'SHA256': query_result_inner[2],  '类型': query_result_inner[5],  '平台': query_result_inner[6],  '家族': query_result_inner[7],'文件拓展名':query_result_inner[10] , '脱壳':query_result_inner[11],'SSDEEP':query_result_inner[12]}
        print(query_result_dict)
        return jsonify({'query_sha256': query_result_dict})
                      
    except pymysql.MySQLError as e:  
        # 如果查询失败（例如表不存在），返回错误信息  
        return jsonify({'error': str(e)}), 500  

@app.route('/download_sha256/<sha256>', methods=['GET'])  
def download_file_sha256(sha256):  
    file_path = get_file_path_and_zip(sha256)  
    if file_path is None:  
        abort(404)
  
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)




if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, threaded=True)