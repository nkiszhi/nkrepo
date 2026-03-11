"""
检测相关API - 完全按照旧后端实现,使用new_flask下的模型
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from app.api.auth import get_current_user
from app.schemas.detect import UploadResponse, DetectionResponse, DomainDetectRequest, DomainDetectResponse
from app.core import settings
import os
import logging
import json
import numpy as np
import hashlib
import shutil
import configparser

router = APIRouter()
logger = logging.getLogger(__name__)

# 导入new_flask下的模型和服务
from app.services.detection.ensemble_predict import run_ensemble_prediction
from app.services.detection.dga_detection import MultiModelDetection

# 导入旧后端的数据库查询工具(因为数据库逻辑在旧代码里)
from app.utils.flask_mysql import Databaseoperation

db_op = Databaseoperation()
dga_detector = None  # 延迟初始化

def get_dga_detector():
    """获取DGA检测器(延迟初始化)"""
    global dga_detector
    if dga_detector is None:
        try:
            dga_detector = MultiModelDetection()
        except Exception as e:
            logger.warning(f"DGA检测器初始化失败: {str(e)}")
    return dga_detector


def convert_to_serializable(obj):
    """转换对象为可序列化格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def get_field_value(result, field_name, index, default=''):
    """从查询结果中获取字段值"""
    try:
        if isinstance(result, dict):
            return result.get(field_name, default)
        elif isinstance(result, (list, tuple)) and len(result) > index:
            return result[index] if result[index] is not None else default
        return default
    except Exception:
        return default


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """文件上传和检测 - 完全按照旧后端实现"""
    try:
        # 保存文件
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 获取文件大小
        file_size_bytes = os.path.getsize(file_path)
        file_size_kb = file_size_bytes / 1024
        file_size = format(file_size_kb, '.2f') + " KB"
        
        # 运行检测模型
        exe_result = run_ensemble_prediction(file_path)
        exe_result = convert_to_serializable(exe_result)
        
        # 查询数据库 - 使用旧后端的filesha256方法
        query_result = db_op.filesha256(file.filename)
        
        if not query_result or query_result == (None, None, '0'):
            logger.error(f"数据库查询结果为空: filename={file.filename}")
            # 返回基础信息
            return {
                'original_filename': file.filename,
                'query_result': {'MD5': '', 'SHA256': ''},
                'file_size': file_size,
                'exe_result': exe_result,
                'VT_API': '',
                'sha256': ''
            }
        
        # 解析查询结果
        if isinstance(query_result, tuple) and len(query_result) == 3:
            str_sha256, str_md5, has_vt = query_result
            
            # 如果数据库有记录,查询详细信息
            if has_vt != '0':
                # 查询完整信息
                detailed_result = db_op.mysqlsha256(str_sha256)
                if detailed_result:
                    detailed_result = convert_to_serializable(detailed_result)
                    if isinstance(detailed_result, (list, tuple)) and len(detailed_result) > 0:
                        detailed_result = detailed_result[0]
                    
                    query_result_dict = {
                        'MD5': get_field_value(detailed_result, 'md5', 3, str_md5),
                        'SHA-256': str_sha256,
                        'SSDEEP': get_field_value(detailed_result, 'ssdeep', 4, ''),
                        'vhash': get_field_value(detailed_result, 'vhash', 5, ''),
                        'Authentihash': get_field_value(detailed_result, 'authentihash', 6, ''),
                        'Imphash': get_field_value(detailed_result, 'imphash', 7, ''),
                        'Rich header hash': get_field_value(detailed_result, 'rich_header_hash', 8, ''),
                        '类型': get_field_value(detailed_result, 'category', 11, ''),
                        '平台': get_field_value(detailed_result, 'platform', 12, ''),
                        '家族': get_field_value(detailed_result, 'family', 13, '')
                    }
                else:
                    query_result_dict = {'MD5': str_md5, 'SHA256': str_sha256}
            else:
                query_result_dict = {'MD5': str_md5, 'SHA256': str_sha256}
            
            VT_API = str_sha256
            logger.info(f"文件上传成功: filename={file.filename}, sha256={str_sha256}")
            
            return {
                'original_filename': file.filename,
                'query_result': query_result_dict,
                'file_size': file_size,
                'exe_result': exe_result,
                'VT_API': VT_API,
                'sha256': str_sha256
            }
        else:
            logger.error(f"数据库查询结果格式错误: {query_result}")
            return {
                'original_filename': file.filename,
                'query_result': {'MD5': '', 'SHA256': ''},
                'file_size': file_size,
                'exe_result': exe_result,
                'VT_API': '',
                'sha256': ''
            }
        
    except Exception as e:
        logger.error(f"文件上传接口异常: {str(e)}")
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.post("/api/detect_url")
async def detect_domain(request: DomainDetectRequest, current_user: dict = Depends(get_current_user)):
    """域名检测 - 完全按照旧Flask格式返回"""
    try:
        url = request.url
        
        # 使用DGA检测器
        detector = get_dga_detector()
        if detector:
            result_tuple = detector.multi_predict_single_dname(url)
            
            # 旧Flask返回的是 (result_dict, status_code) 元组
            if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                result_dict, status_code = result_tuple
                # 转换numpy类型为Python原生类型
                result_dict = convert_numpy_to_python(result_dict)
                return {
                    'status': '1' if status_code else '0',
                    'result': result_dict
                }
            else:
                # 如果返回格式不对,返回错误
                return {'error': 'Unexpected result format'}
        else:
            # 如果检测器初始化失败,返回错误
            return {'error': 'DGA detector not initialized'}
            
    except Exception as e:
        logger.error(f"域名检测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


def convert_numpy_to_python(obj):
    """转换numpy类型为Python原生类型 - 与旧Flask完全一致"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(i) for i in obj]
    elif isinstance(obj, (np.int64, np.float64)):
        return obj.item()
    else:
        return obj


@router.get("/detection_API/{sha256}")
async def get_detection_API(sha256: str, current_user: dict = Depends(get_current_user)):
    """VT检测查询 - 完全按照旧Flask实现"""
    try:
        # 验证SHA256
        if not sha256 or sha256.lower() == 'undefined' or len(sha256) != 64:
            logger.error(f"无效的SHA256值: {sha256}")
            raise HTTPException(status_code=400, detail="无效的SHA256值（长度必须为64位）")
        
        # 从数据库读取VT API密钥
        from app.scripts.config_manager import ConfigManager
        cm = ConfigManager()
        api_key = cm.get_config('api', 'vt_key', '')
        
        if not api_key:
            logger.error("VT API密钥未配置")
            raise HTTPException(status_code=500, detail="VT API密钥未配置")
        
        # 查找样本文件
        sample_dir_path, sample_file_path = get_existing_sample_path(sha256)
        if not sample_dir_path or not sample_file_path:
            logger.error(f"样本文件不存在: sha256={sha256}")
            raise HTTPException(status_code=404, detail="样本文件不存在")
        
        # 导入VT API
        from app.services.external.api_vt import VT
        
        # 调用VT API
        json_file_path = VT.get_API_result_detection(sha256, api_key, sample_dir_path, sample_file_path)
        
        if json_file_path == 500:
            raise HTTPException(status_code=500, detail="VT检测服务失败")
        
        # 更新数据库
        db_op.update_db(sha256)
        
        # 读取结果
        with open(json_file_path, 'r') as f:
            scan_result = json.load(f)
        
        if 'data' not in scan_result or 'attributes' not in scan_result['data']:
            raise HTTPException(status_code=400, detail="Missing required data in JSON file")
        
        results = []
        last_analysis_results = scan_result['data']['attributes'].get(
            'last_analysis_results',
            scan_result['data']['attributes'].get('results', {})
        )
        for engine in last_analysis_results.values():
            results.append({
                "method": engine.get('method', ''),
                "engine_name": engine.get('engine_name', ''),
                "engine_version": engine.get('engine_version', ''),
                "engine_update": engine.get('engine_update', ''),
                "category": engine.get('category', ''),
                "result": engine.get('result', '')
            })
        
        logger.info(f"VT detection查询成功: sha256={sha256}")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"VT detection API异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


def get_existing_sample_path(sha256):
    """查找样本文件路径 - 按照旧后端实现"""
    try:
        prefix_parts = [sha256[0], sha256[1], sha256[2], sha256[3], sha256[4]]
        old_sample_dir = os.path.join('../../../data/samples', *prefix_parts)
        old_sample_path = os.path.join(old_sample_dir, sha256)
        new_sample_dir = '../../../data/web_upload_file'
        new_sample_path = os.path.join(new_sample_dir, sha256)
        
        if os.path.exists(old_sample_path):
            return old_sample_dir, old_sample_path
        elif os.path.exists(new_sample_path):
            return new_sample_dir, new_sample_path
        else:
            return None, None
    except Exception as e:
        logger.error(f"查找样本路径异常: {str(e)}")
        return None, None


@router.post("/detect_by_sha256")
async def detect_by_sha256(request: dict, current_user: dict = Depends(get_current_user)):
    """SHA256检测 - 按照旧后端实现"""
    try:
        sha256 = request.get('sha256', '').strip().lower()
        
        if not sha256 or len(sha256) != 64:
            raise HTTPException(status_code=400, detail="无效的SHA256值")
        
        logger.info(f"SHA256检测请求: {sha256}")
        
        # 查询数据库
        query_result = db_op.mysqlsha256s(sha256)
        
        if not query_result:
            raise HTTPException(status_code=404, detail="数据库中没有该SHA256对应的样本")
        
        query_result = convert_to_serializable(query_result)
        query_result_inner = query_result[0] if isinstance(query_result, (list, tuple)) else query_result
        
        filename = get_field_value(query_result_inner, 'name', 0, sha256)
        
        query_result_dict = {
            'MD5': get_field_value(query_result_inner, 'md5', 3, ''),
            'SHA-256': sha256,
            'SSDEEP': get_field_value(query_result_inner, 'ssdeep', 4, ''),
            'vhash': get_field_value(query_result_inner, 'vhash', 5, ''),
            'Authentihash': get_field_value(query_result_inner, 'authentihash', 6, ''),
            'Imphash': get_field_value(query_result_inner, 'imphash', 7, ''),
            'Rich header hash': get_field_value(query_result_inner, 'rich_header_hash', 8, ''),
            '类型': get_field_value(query_result_inner, 'category', 11, ''),
            '平台': get_field_value(query_result_inner, 'platform', 12, ''),
            '家族': get_field_value(query_result_inner, 'family', 13, ''),
            '文件大小': f"{get_field_value(query_result_inner, 'length', 1, 0)} bytes",
        }
        
        # 尝试获取模型检测结果
        exe_result = {}
        try:
            sample_dir_path, sample_file_path = get_existing_sample_path(sha256)
            if sample_file_path and os.path.exists(sample_file_path):
                exe_result = run_ensemble_prediction(sample_file_path)
                exe_result = convert_to_serializable(exe_result)
        except Exception as model_error:
            exe_result = {'error': str(model_error)}
        
        return {
            'success': True,
            'original_filename': filename,
            'query_result': query_result_dict,
            'file_size': query_result_dict.get('文件大小', '未知'),
            'exe_result': exe_result,
            'VT_API': sha256,
            'sha256': sha256,
            'detection_mode': 'sha256'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SHA256检测接口异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get("/behaviour_API/{sha256}")
async def get_behaviour_API(sha256: str, current_user: dict = Depends(get_current_user)):
    """VT行为查询 - 完全按照旧Flask实现"""
    try:
        # 验证SHA256
        if not sha256 or sha256.lower() == 'undefined' or len(sha256) != 64:
            logger.error(f"无效的SHA256值: {sha256}")
            raise HTTPException(status_code=400, detail="无效的SHA256值（长度必须为64位）")
        
        # 从数据库读取VT API密钥
        from app.scripts.config_manager import ConfigManager
        cm = ConfigManager()
        api_key = cm.get_config('api', 'vt_key', '')
        
        if not api_key:
            logger.error("VT API密钥未配置")
            raise HTTPException(status_code=500, detail="VT API密钥未配置")
        
        # 查找样本文件
        sample_dir_path, sample_file_path = get_existing_sample_path(sha256)
        if not sample_dir_path or not sample_file_path:
            logger.error(f"样本文件不存在: sha256={sha256}")
            raise HTTPException(status_code=404, detail="样本文件不存在")
        
        # 导入VT API
        from app.services.external.api_vt import VT
        
        # 调用VT API
        behaviour_file_path = VT.get_API_result_behaviour(sha256, api_key, sample_dir_path, sample_file_path)
        
        with open(behaviour_file_path, 'r') as f:
            behaviour_scan = json.load(f)
            behaviour_data = behaviour_scan.get('data', {})
        
        logger.info(f"VT behaviour查询成功: sha256={sha256}")
        return behaviour_data
        
    except HTTPException:
        raise
    except json.JSONDecodeError:
        logger.error(f"Behaviour文件JSON解析错误: sha256={sha256}")
        raise HTTPException(status_code=400, detail="Error decoding JSON file")
    except Exception as e:
        logger.error(f"VT behaviour API异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
