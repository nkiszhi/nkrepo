"""
查询相关API - 完全兼容旧后端路由
"""
from fastapi import APIRouter, HTTPException, Depends
from app.api.auth import get_current_user
from app.core import db
from app.utils.flask_mysql import Databaseoperation
import logging
import sys
import os

router = APIRouter()
logger = logging.getLogger(__name__)

db_op = Databaseoperation()


def convert_to_serializable(obj):
    """转换对象为可序列化格式"""
    import numpy as np
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


@router.post("/query_sha256")
async def query_sha256(request: dict, current_user: dict = Depends(get_current_user)):
    """SHA256查询 - 完全按照旧Flask实现"""
    try:
        # 旧Flask使用tableName字段
        sha256 = request.get('tableName', '').strip().lower()
        
        logger.info(f"SHA256查询请求: 传递的sha256='{sha256}'")
        
        if not sha256:
            logger.warning("SHA256查询失败：未提供sha256值")
            return {
                'success': False,
                'error': '未提供SHA256值',
                'data': {}
            }
        
        # 验证SHA256格式
        if len(sha256) != 64:
            logger.warning(f"SHA256格式错误：长度={len(sha256)}, 值='{sha256}'")
            return {
                'success': False,
                'error': '无效的SHA256值（长度必须为64位）',
                'data': {}
            }
        
        # 查询数据库
        query_result = db_op.mysqlsha256s(sha256)
        logger.info(f"数据库查询结果: sha256='{sha256}', 结果={query_result}")
        
        if not query_result:
            logger.warning(f"未找到记录: sha256='{sha256}'")
            return {
                'success': False,
                'error': '未找到该SHA256对应的记录',
                'data': {}
            }
        
        query_result = convert_to_serializable(query_result)
        
        # 兼容列表/单个字典两种返回格式
        query_result_inner = query_result[0] if isinstance(query_result, (list, tuple)) and len(query_result) > 0 else query_result
        
        # 构建详情字典 - 完全按照旧Flask格式
        detail_dict = {
            'MD5': get_field_value(query_result_inner, 'md5', 3, ''),
            'SHA256': get_field_value(query_result_inner, 'sha256', 2, ''),
            'SSDEEP': get_field_value(query_result_inner, 'ssdeep', 4, ''),
            'vhash': get_field_value(query_result_inner, 'vhash', 5, ''),
            'Authentihash': get_field_value(query_result_inner, 'authentihash', 6, ''),
            'Imphash': get_field_value(query_result_inner, 'imphash', 7, ''),
            'Rich header hash': get_field_value(query_result_inner, 'rich_header_hash', 8, ''),
            '类型': get_field_value(query_result_inner, 'category', 11, ''),
            '平台': get_field_value(query_result_inner, 'platform', 12, ''),
            '家族': get_field_value(query_result_inner, 'family', 13, '')
        }
        
        logger.info(f"SHA256查询成功: sha256='{sha256}'")
        return {
            'success': True,
            'error': '',
            'data': detail_dict
        }
        
    except Exception as e:
        logger.error(f"SHA256查询失败: {str(e)}")
        return {
            'success': False,
            'error': f'查询失败: {str(e)}',
            'data': {}
        }


@router.post("/query_family")
async def query_family(request: dict, current_user: dict = Depends(get_current_user)):
    """家族查询 - 完全按照旧Flask实现"""
    try:
        # 旧Flask使用tableName字段
        short_table_name = request.get('tableName', None)
        
        if not short_table_name:
            return {'error': '未提供类型名称'}
        
        # 自动拼接正确前缀：family_ + 前端传递的短表名
        full_table_name = f'family_{short_table_name}'
        
        # 从config.ini读取家族数据库名
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        db_family = config.get('mysql', 'db_family', fallback='family')
        
        # 查询数据库 - 使用family数据库
        sha256s = custom_mysql_query(full_table_name, db_family)
        
        if not sha256s:
            logger.warning(f"未找到family相关数据: table={full_table_name}")
            return {'error': '未找到相关数据'}
        
        logger.info(f"家族查询成功: table={full_table_name}, 结果数量={len(sha256s)}")
        return {'sha256s': sha256s}
        
    except Exception as e:
        logger.error(f"查询family异常: {str(e)}")
        return {'error': f'服务器内部错误: {str(e)}'}


def custom_mysql_query(table_name: str, db_name: str, limit: int = 20):
    """
    自定义数据库查询 - 完全按照旧Flask实现
    直接连接指定数据库执行查询,获取随机20个sha256
    """
    import pymysql
    import configparser
    import re as regex
    
    conn = None
    cursor = None
    try:
        # 读取数据库配置
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        conn = pymysql.connect(
            host=config.get('mysql', 'host'),
            user=config.get('mysql', 'user'),
            password=config.get('mysql', 'passwd'),
            database=db_name,
            charset=config.get('mysql', 'charset', fallback='utf8'),
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = conn.cursor()
        
        # 验证表名合法性（防止SQL注入）
        if not table_name or not isinstance(table_name, str) or len(table_name) > 100:
            raise ValueError(f"非法表名: {table_name}")
        # 只允许字母、数字、下划线（符合MySQL表名规范）
        if not regex.match(r'^[a-zA-Z0-9_]+$', table_name):
            raise ValueError(f"表名包含非法字符: {table_name}")
        
        # 执行查询：随机获取20个sha256
        query_sql = f"""
            SELECT sha256 FROM `{table_name}` 
            ORDER BY RAND() 
            LIMIT {limit}
        """
        cursor.execute(query_sql)
        results = cursor.fetchall()
        
        # 提取sha256列表
        sha256_list = [item.get('sha256', '').strip() for item in results if item.get('sha256')]
        sha256_list = list(filter(None, sha256_list))  # 过滤空值
        
        logger.info(f"自定义查询成功：db={db_name}, table={table_name}, 找到{len(sha256_list)}个sha256")
        return sha256_list
        
    except Exception as e:
        logger.error(f"自定义查询失败：db={db_name}, table={table_name}, error={str(e)}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.post("/query_platform")
async def query_platform(request: dict, current_user: dict = Depends(get_current_user)):
    """平台查询 - 完全按照旧Flask实现"""
    try:
        # 旧Flask使用tableName字段
        short_table_name = request.get('tableName', None)
        
        if not short_table_name:
            return {'error': '未提供类型名称'}
        
        # 自动拼接正确前缀：platform_ + 前端传递的短表名
        full_table_name = f'platform_{short_table_name}'
        
        # 从config.ini读取平台数据库名
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        db_platform = config.get('mysql', 'db_platform', fallback='platform')
        
        # 查询数据库 - 使用platform数据库
        sha256s = custom_mysql_query(full_table_name, db_platform)
        
        if not sha256s:
            logger.warning(f"未找到platform相关数据: table={full_table_name}")
            return {'error': '未找到相关数据'}
        
        logger.info(f"平台查询成功: table={full_table_name}, 结果数量={len(sha256s)}")
        return {'sha256s': sha256s}
        
    except Exception as e:
        logger.error(f"查询platform异常: {str(e)}")
        return {'error': f'服务器内部错误: {str(e)}'}

        
        # 执行查询：随机获取20个sha256
        query_sql = f"""
            SELECT sha256 FROM `{table_name}` 
            ORDER BY RAND() 
            LIMIT {limit}
        """
        cursor.execute(query_sql)
        results = cursor.fetchall()
        
        # 提取sha256列表
        sha256_list = [item.get('sha256', '').strip() for item in results if item.get('sha256')]
        sha256_list = list(filter(None, sha256_list))  # 过滤空值
        
        logger.info(f"自定义查询成功：db={db_name}, table={table_name}, 找到{len(sha256_list)}个sha256")
        return sha256_list
        
    except Exception as e:
        logger.error(f"自定义查询失败：db={db_name}, table={table_name}, error={str(e)}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.post("/query_category")
async def query_category(request: dict, current_user: dict = Depends(get_current_user)):
    """分类查询 - 完全按照旧Flask实现"""
    try:
        # 旧Flask使用tableName字段
        short_table_name = request.get('tableName', None)
        
        if not short_table_name:
            return {'error': '未提供类型名称'}
        
        # 自动拼接正确前缀：category_ + 前端传递的短表名
        full_table_name = f'category_{short_table_name}'
        
        # 从config.ini读取分类数据库名
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        db_category = config.get('mysql', 'db_category', fallback='category')
        
        # 查询数据库 - 使用category数据库
        sha256s = custom_mysql_query(full_table_name, db_category)
        
        if not sha256s:
            logger.warning(f"未找到category相关数据: table={full_table_name}")
            return {'error': '未找到相关数据'}
        
        logger.info(f"分类查询成功: table={full_table_name}, 结果数量={len(sha256s)}")
        return {'sha256s': sha256s}
        
    except Exception as e:
        logger.error(f"查询category异常: {str(e)}")
        return {'error': f'服务器内部错误: {str(e)}'}


@router.post("/query_platform")
async def query_platform(request: dict, current_user: dict = Depends(get_current_user)):
    """平台查询 - 完全按照旧Flask实现"""
    try:
        # 旧Flask使用tableName字段
        short_table_name = request.get('tableName', None)
        
        if not short_table_name:
            return {'error': '未提供类型名称'}
        
        # 自动拼接正确前缀：platform_ + 前端传递的短表名
        full_table_name = f'platform_{short_table_name}'
        
        # 从config.ini读取平台数据库名
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        db_platform = config.get('mysql', 'db_platform', fallback='platform')
        
        # 查询数据库 - 使用platform数据库
        sha256s = custom_mysql_query(full_table_name, db_platform)
        
        if not sha256s:
            logger.warning(f"未找到platform相关数据: table={full_table_name}")
            return {'error': '未找到相关数据'}
        
        logger.info(f"平台查询成功: table={full_table_name}, 结果数量={len(sha256s)}")
        return {'sha256s': sha256s}
        
    except Exception as e:
        logger.error(f"查询platform异常: {str(e)}")
        return {'error': f'服务器内部错误: {str(e)}'}




@router.get("/detail_category/{sha256}")
async def detail_category(sha256: str, current_user: dict = Depends(get_current_user)):
    """分类详情 - 完全按照旧Flask实现"""
    result, status_code = get_detail_common(sha256)
    return result


@router.get("/detail_family/{sha256}")
async def detail_family(sha256: str, current_user: dict = Depends(get_current_user)):
    """家族详情 - 完全按照旧Flask实现"""
    result, status_code = get_detail_common(sha256)
    return result


@router.get("/detail_platform/{sha256}")
async def detail_platform(sha256: str, current_user: dict = Depends(get_current_user)):
    """平台详情 - 完全按照旧Flask实现"""
    result, status_code = get_detail_common(sha256)
    return result


def get_detail_common(sha256: str):
    """通用详情处理 - 完全按照旧Flask实现"""
    try:
        # 校验SHA256格式
        if not sha256 or len(sha256) != 64:
            return {
                'success': False,
                'error': '无效的SHA256值（长度必须为64位）',
                'data': {}
            }, 400
        
        query_result = db_op.mysqlsha256s(sha256)
        logger.info(f"SHA256查询数据库返回: sha256={sha256}, 原始结果={query_result}")
        
        if not query_result:
            return {
                'success': False,
                'error': '未找到相关记录',
                'data': {}
            }, 404
        
        query_result = convert_to_serializable(query_result)
        # 兼容列表/单个字典两种返回格式
        query_result_inner = query_result[0] if isinstance(query_result, (list, tuple)) and len(query_result) > 0 else query_result
        
        # 构建详情字典
        detail_dict = {
            'MD5': get_field_value(query_result_inner, 'md5', 3, ''),
            'SHA256': get_field_value(query_result_inner, 'sha256', 2, ''),
            'SSDEEP': get_field_value(query_result_inner, 'ssdeep', 4, ''),
            'vhash': get_field_value(query_result_inner, 'vhash', 5, ''),
            'Authentihash': get_field_value(query_result_inner, 'authentihash', 6, ''),
            'Imphash': get_field_value(query_result_inner, 'imphash', 7, ''),
            'Rich header hash': get_field_value(query_result_inner, 'rich_header_hash', 8, ''),
            '类型': get_field_value(query_result_inner, 'category', 11, ''),
            '平台': get_field_value(query_result_inner, 'platform', 12, ''),
            '家族': get_field_value(query_result_inner, 'family', 13, '')
        }
        
        logger.info(f"详情查询成功: sha256={sha256}")
        return {
            'success': True,
            'error': '',
            'data': detail_dict
        }, 200
        
    except Exception as e:
        logger.error(f"详情查询异常: sha256={sha256}, error={str(e)}")
        return {
            'success': False,
            'error': f'服务器内部错误: {str(e)}',
            'data': {}
        }, 500


# ==================== 下载功能 ====================

@router.get("/download_category/{sha256}")
async def download_category(sha256: str, current_user: dict = Depends(get_current_user)):
    """分类下载 - 完全按照旧Flask实现"""
    file_path = get_file_path_and_zip(sha256)
    if file_path is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        file_path,
        filename=os.path.basename(file_path),
        media_type='application/zip'
    )


@router.get("/download_family/{sha256}")
async def download_family(sha256: str, current_user: dict = Depends(get_current_user)):
    """家族下载 - 完全按照旧Flask实现"""
    file_path = get_file_path_and_zip(sha256)
    if file_path is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        file_path,
        filename=os.path.basename(file_path),
        media_type='application/zip'
    )


@router.get("/download_platform/{sha256}")
async def download_platform(sha256: str, current_user: dict = Depends(get_current_user)):
    """平台下载 - 完全按照旧Flask实现"""
    file_path = get_file_path_and_zip(sha256)
    if file_path is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        file_path,
        filename=os.path.basename(file_path),
        media_type='application/zip'
    )


@router.get("/download_sha256/{sha256}")
async def download_sha256(sha256: str, current_user: dict = Depends(get_current_user)):
    """SHA256下载 - 完全按照旧Flask实现"""
    file_path = get_file_path_and_zip(sha256)
    if file_path is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        file_path,
        filename=os.path.basename(file_path),
        media_type='application/zip'
    )


def get_file_path_and_zip(sha256: str, zip_password: str = "infected"):
    """
    获取文件路径并压缩 - 完全按照旧Flask实现
    
    样本文件查找路径:
    1. ../../../data/samples/{sha256[0]}/{sha256[1]}/{sha256[2]}/{sha256[3]}/{sha256[4]}/{sha256}
       例如: data/samples/6/a/8/3/9/6a839df92dc95ef2739d0e2501a09f9b29b8a0141c0446308b6db1797554528c
    2. ../../../data/web_upload_file/{sha256}
       例如: data/web_upload_file/6a839df92dc95ef2739d0e2501a09f9b29b8a0141c0446308b6db1797554528c
    
    压缩文件保存路径:
    - ../../../data/zips/{sha256}.zip
    """
    import subprocess
    
    try:
        # 校验SHA256格式
        if not sha256 or len(sha256) != 64:
            logger.error(f"无效的SHA256值: {sha256}")
            return None
        
        # 提取前5个字符作为五级目录
        prefix = sha256[:5]
        
        # 样本文件路径1: data/samples/{五级目录}/{sha256}
        original_sample_path = os.path.join('../../../data/samples', *prefix, sha256)
        
        # 样本文件路径2: data/web_upload_file/{sha256}
        web_upload_path = os.path.join('../../../data/web_upload_file', sha256)
        
        # 压缩文件路径: data/zips/{sha256}.zip
        zip_file_path = os.path.join('../../../data/zips', sha256 + '.zip')
        
        # 检查样本文件是否存在
        if os.path.exists(original_sample_path):
            target_file_path = original_sample_path
            logger.info(f"找到样本文件(原始路径): {original_sample_path}")
        elif os.path.exists(web_upload_path):
            target_file_path = web_upload_path
            logger.info(f"找到样本文件(上传路径): {web_upload_path}")
        else:
            logger.warning(f"样本文件不存在，无法压缩: sha256={sha256}")
            return None
        
        # 如果压缩文件已存在,直接返回
        if os.path.exists(zip_file_path):
            logger.info(f"压缩文件已存在: {zip_file_path}")
            return zip_file_path
        
        # 创建zips目录
        os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)
        
        # 使用7z压缩文件
        command = [
            '7z', 'a', '-tzip', f'-p{zip_password}',
            zip_file_path, target_file_path
        ]
        
        # 执行压缩
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"文件压缩成功: sha256={sha256}, zip_path={zip_file_path}")
        
        return zip_file_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"压缩文件失败: sha256={sha256}, error={e.stderr}")
        return None
    except Exception as e:
        logger.error(f"压缩文件异常: {str(e)}")
        return None
