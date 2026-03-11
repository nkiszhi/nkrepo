"""
ATT&CK矩阵完整API - 从旧Flask完整迁移
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from app.api.auth import get_current_user
from app.core import db
from app.scripts.config_manager import ConfigManager
import logging
import pymysql
import pymysql.cursors

router = APIRouter()
logger = logging.getLogger(__name__)


def get_attck_db_config():
    """获取ATT&CK数据库配置"""
    import configparser
    import os
    
    # 直接读取config.ini
    config_file = os.path.join(os.path.dirname(__file__), '../../config.ini')
    cp = configparser.ConfigParser()
    if os.path.exists(config_file):
        cp.read(config_file, encoding='utf-8')
    
    return {
        'host': cp.get('mysql', 'host', fallback='localhost'),
        'user': cp.get('mysql', 'user', fallback='root'),
        'password': cp.get('mysql', 'passwd', fallback=''),
        'database': cp.get('mysql', 'db_component', fallback='api_component'),
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }


def get_attck_connection():
    """获取ATT&CK数据库连接"""
    config = get_attck_db_config()
    logger.info(f"连接ATT&CK数据库: {config['host']}/{config['database']}")
    return pymysql.connect(**config)


# ==================== 核心查询函数 ====================

def query_attck_technique_list(page=1, page_size=20, search=None, technique_id=None):
    """查询ATT&CK技术列表"""
    conn = None
    cursor = None
    try:
        print(f"[DEBUG] 开始查询ATT&CK技术列表: technique_id={technique_id}")
        conn = get_attck_connection()
        cursor = conn.cursor()
        print(f"[DEBUG] 数据库连接成功")
        
        # 基础查询：关联主表和技术表
        base_sql = """
            SELECT mi.file_name, mi.status, mi.root_function, mi.alias, mi.summary,
                   mi.source_hlil, mi.generated_cpp, mi.tries, mi.created_at, mi.updated_at,
                   ti.technique_id, ti.technique_name, ti.confidence,
                   tt.tactic_id, tt.tactic_name, tt.Description as tactic_description
            FROM main_info mi
            LEFT JOIN technique_info ti ON mi.file_name = ti.file_name AND mi.alias = ti.alias
            LEFT JOIN tactics_technique tt ON ti.technique_id = tt.technique_id
        """
        
        # 条件和参数
        conditions = []
        params = []
        
        if search:
            conditions.append("(mi.alias LIKE %s OR mi.root_function LIKE %s OR ti.technique_id LIKE %s OR ti.technique_name LIKE %s)")
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param, search_param])
        
        if technique_id:
            conditions.append("ti.technique_id = %s")
            params.append(technique_id)
        
        # 拼接WHERE子句
        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)

        # 总条数查询 - 直接使用COUNT
        count_sql = base_sql.replace("SELECT mi.file_name, mi.status, mi.root_function, mi.alias, mi.summary,\n                   mi.source_hlil, mi.generated_cpp, mi.tries, mi.created_at, mi.updated_at,\n                   ti.technique_id, ti.technique_name, ti.confidence,\n                   tt.tactic_id, tt.tactic_name, tt.Description as tactic_description", "SELECT COUNT(DISTINCT mi.file_name) as total")
        cursor.execute(count_sql, params)
        total = cursor.fetchone()['total']

        # 分页处理
        offset = (page - 1) * page_size
        final_sql = f"{base_sql} GROUP BY mi.file_name, mi.status, mi.root_function, mi.alias, mi.summary, mi.source_hlil, mi.generated_cpp, mi.tries, mi.created_at, mi.updated_at, ti.technique_id, ti.technique_name, ti.confidence, tt.tactic_id, tt.tactic_name, tt.Description LIMIT %s OFFSET %s"
        params.extend([page_size, offset])
        
        cursor.execute(final_sql, params)
        results = cursor.fetchall()
        
        # 处理子函数别名
        result_list = []
        for item in results:
            file_name = item['file_name']
            alias = item['alias']
            
            # 查询子函数别名
            child_sql = """
                SELECT children_aliases_key, children_aliases_value 
                FROM children_aliases_info 
                WHERE file_name = %s AND alias = %s
            """
            cursor.execute(child_sql, [file_name, alias])
            children = cursor.fetchall()
            
            # 构建返回数据
            technique_item = {
                "file_name": item['file_name'],
                "status": item['status'],
                "root_function": item['root_function'],
                "alias": item['alias'],
                "summary": item['summary'],
                "source_hlil": item['source_hlil'],
                "generated_cpp": item['generated_cpp'],
                "tries": item['tries'],
                "created_at": item['created_at'].isoformat() if item['created_at'] else None,
                "updated_at": item['updated_at'].isoformat() if item['updated_at'] else None,
                "techniques": [{
                    "technique_id": item['technique_id'],
                    "technique_name": item['technique_name'],
                    "confidence": item['confidence'],
                    "tactic_id": item['tactic_id'],
                    "tactic_name": item['tactic_name'],
                    "tactic_description": item['tactic_description']
                }] if item['technique_id'] else [],
                "children_aliases": {
                    child['children_aliases_key']: child['children_aliases_value'] 
                    for child in children
                }
            }
            result_list.append(technique_item)
        
        return {
            "success": True,
            "data": result_list,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size
            }
        }
        
    except pymysql.MySQLError as e:
        logger.error(f"ATT&CK技术列表查询异常: {str(e)}")
        return {"success": False, "error": f"数据库错误: {str(e)}", "data": [], "pagination": {}}
    except Exception as e:
        logger.error(f"ATT&CK技术列表查询通用异常: {str(e)}")
        return {"success": False, "error": f"服务器内部错误: {str(e)}", "data": [], "pagination": {}}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def query_attck_technique_detail(file_name, alias):
    """查询ATT&CK技术详情"""
    conn = None
    cursor = None
    try:
        conn = get_attck_connection()
        cursor = conn.cursor()
        
        # 查询主信息
        main_sql = """
            SELECT mi.file_name, mi.status, mi.root_function, mi.alias, mi.summary,
                   mi.source_hlil, mi.alias_map_json, mi.generated_cpp, mi.cpp_code,
                   mi.tries, mi.technique_count, mi.children_aliases_count,
                   mi.created_at, mi.updated_at
            FROM main_info mi
            WHERE mi.file_name = %s AND mi.alias = %s
        """
        cursor.execute(main_sql, [file_name, alias])
        main_info = cursor.fetchone()
        
        if not main_info:
            return {"success": False, "error": "未找到该技术信息", "data": None}
        
        # 查询关联的技术
        tech_sql = """
            SELECT ti.technique_id, ti.technique_name, ti.confidence,
                   tt.tactic_id, tt.tactic_name, tt.Description as tactic_description
            FROM technique_info ti
            LEFT JOIN tactics_technique tt ON ti.technique_id = tt.technique_id
            WHERE ti.file_name = %s AND ti.alias = %s
        """
        cursor.execute(tech_sql, [file_name, alias])
        techniques = cursor.fetchall()
        
        # 查询子函数别名
        child_sql = """
            SELECT children_aliases_key, children_aliases_value 
            FROM children_aliases_info 
            WHERE file_name = %s AND alias = %s
        """
        cursor.execute(child_sql, [file_name, alias])
        children = cursor.fetchall()
        
        # 构建返回数据
        result = {
            "id": 0,  # 详情页不需要显示序号
            "hash_id": main_info['file_name'],
            "api_component": main_info['alias'],
            "status": main_info['status'],
            "root_function": main_info['root_function'],
            "summary": main_info['summary'],
            "source_hlil": main_info['source_hlil'],
            "alias_map_json": main_info.get('alias_map_json', ''),
            "generated_cpp": main_info['generated_cpp'],
            "cpp_code": main_info.get('cpp_code', ''),
            "tries": main_info['tries'],
            "technique_count": main_info.get('technique_count', 0),
            "children_aliases_count": main_info.get('children_aliases_count', 0),
            "created_at": main_info['created_at'].isoformat() if main_info['created_at'] else None,
            "updated_at": main_info['updated_at'].isoformat() if main_info['updated_at'] else None,
            "techniques": techniques,
            "children_aliases": {
                child['children_aliases_key']: child['children_aliases_value']
                for child in children
            }
        }
        
        return {"success": True, "data": result}
        
    except Exception as e:
        logger.error(f"ATT&CK技术详情查询异常: {str(e)}")
        return {"success": False, "error": f"服务器内部错误: {str(e)}", "data": None}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== API路由 ====================

@router.get("/dev-api/api/attck/matrix")
async def get_attck_matrix(current_user: dict = Depends(get_current_user)):
    """获取ATT&CK矩阵数据（适配前端展示）"""
    conn = None
    cursor = None
    try:
        conn = get_attck_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            tt.tactic_id, tt.tactic_name,
            tt.technique_id, tt.technique_name, tt.Description as description
        FROM 
            tactics_technique tt
        WHERE 
            tt.technique_id IS NOT NULL AND tt.technique_id != ''
        ORDER BY 
            tt.tactic_id, tt.technique_id
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # 构建矩阵数据结构
        matrix_data = {}
        
        for item in results:
            tactic_id = item['tactic_id']
            tactic_name = item['tactic_name']
            
            if tactic_id not in matrix_data:
                matrix_data[tactic_id] = {
                    "tactic_id": tactic_id,
                    "tactic_name": tactic_name,
                    "techniques": []
                }
            
            # 只添加有效技术
            if item.get('technique_id'):
                matrix_data[tactic_id]['techniques'].append({
                    "technique_id": item.get('technique_id', ''),
                    "technique_name": item.get('technique_name', ''),
                    "description": item.get('description', '') or '暂无描述'
                })
        
        # 转换为列表并去重
        result_list = [v for k, v in matrix_data.items() if v['techniques']]
        
        logger.info(f"ATT&CK矩阵数据查询成功：{len(result_list)}个战术，{len(results)}个技术")
        return {
            "success": True,
            "data": result_list
        }
    except Exception as e:
        logger.error(f"ATT&CK矩阵接口异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.get("/dev-api/api/attck/matrix/stats")
async def get_matrix_stats_light(current_user: dict = Depends(get_current_user)):
    """获取ATT&CK矩阵统计"""
    conn = None
    cursor = None
    try:
        conn = get_attck_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            technique_id,
            technique_name,
            COUNT(*) as component_count
        FROM technique_info
        GROUP BY technique_id, technique_name
        ORDER BY technique_id
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # 构建function_stats
        function_stats = {}
        for row in results:
            function_stats[row['technique_id']] = row['component_count']
        
        return {
            'success': True,
            'function_stats': function_stats,
            'count': len(function_stats)
        }
        
    except Exception as e:
        logger.warning(f"矩阵统计查询失败: {str(e)}")
        return {
            'success': True,
            'function_stats': {},
            'count': 0,
            'message': f'查询失败: {str(e)}'
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.get("/dev-api/api/attck/function/list")
async def get_function_list(
    technique_id: Optional[str] = Query(None, description="技术ID"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页数量"),
    current_user: dict = Depends(get_current_user)
):
    """获取函数列表"""
    logger.info(f"[API] 调用get_function_list: technique_id={technique_id}")
    result = query_attck_technique_list(page, page_size, None, technique_id)

    if not result['success']:
        logger.error(f"[API] 查询失败: {result.get('error')}")
        raise HTTPException(status_code=500, detail=result.get('error', '查询失败'))

    return result


@router.get("/dev-api/api/attck/function/detail")
async def get_function_detail(
    file_name: str = Query(..., description="文件名"),
    alias: str = Query(..., description="别名"),
    current_user: dict = Depends(get_current_user)
):
    """获取函数详情"""
    result = query_attck_technique_detail(file_name, alias)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', '未找到'))
    
    return result


@router.get("/dev-api/api/attck/techniques")
async def get_attck_techniques(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    technique_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """获取ATT&CK技术列表（支持分页、搜索、筛选）"""
    result = query_attck_technique_list(page, page_size, search, technique_id)
    
    if not result['success']:
        raise HTTPException(status_code=500, detail=result.get('error', '查询失败'))
    
    return result


@router.get("/dev-api/api/attck/techniques/{technique_id}")
async def get_attck_technique_detail_by_id(
    technique_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取单个ATT&CK技术详情（通过技术ID）"""
    conn = None
    cursor = None
    try:
        conn = get_attck_connection()
        cursor = conn.cursor()
        
        # 查询技术基本信息
        tech_sql = """
            SELECT 
                tt.technique_id, tt.technique_name, tt.tactic_id, tt.tactic_name,
                tt.Description as description, tt.chinese_description
            FROM 
                tactics_technique tt
            WHERE 
                tt.technique_id = %s
            LIMIT 1
        """
        cursor.execute(tech_sql, [technique_id])
        tech_info = cursor.fetchone()
        
        if not tech_info:
            raise HTTPException(status_code=404, detail="未找到该技术信息")
        
        # 查询关联函数数量
        func_count_sql = """
            SELECT 
                COUNT(DISTINCT CONCAT(ti.file_name, '|', ti.alias)) as function_count
            FROM 
                technique_info ti
            WHERE 
                ti.technique_id = %s
        """
        cursor.execute(func_count_sql, [technique_id])
        func_count_result = cursor.fetchone()
        function_count = func_count_result.get('function_count', 0) if func_count_result else 0
        
        # 查询子技术
        sub_tech_sql = """
            SELECT 
                tt.technique_id, tt.technique_name, tt.chinese_description,
                COUNT(DISTINCT CONCAT(ti.file_name, '|', ti.alias)) as function_count
            FROM 
                tactics_technique tt
            LEFT JOIN 
                technique_info ti ON tt.technique_id = ti.technique_id
            WHERE 
                tt.technique_id LIKE %s AND tt.technique_id != %s
            GROUP BY 
                tt.technique_id, tt.technique_name, tt.chinese_description
        """
        cursor.execute(sub_tech_sql, [f"{technique_id}%", technique_id])
        sub_techniques = cursor.fetchall()
        
        result = {
            "technique_id": tech_info['technique_id'],
            "technique_name": tech_info['technique_name'],
            "tactic_id": tech_info['tactic_id'],
            "tactic_name": tech_info['tactic_name'],
            "description": tech_info['description'],
            "chinese_description": tech_info.get('chinese_description', ''),
            "function_count": function_count,
            "subtechniques": [
                {
                    "technique_id": sub.get('technique_id'),
                    "technique_name": sub.get('technique_name'),
                    "chinese_description": sub.get('chinese_description'),
                    "function_count": sub.get('function_count', 0)
                }
                for sub in sub_techniques
            ]
        }
        
        return {"success": True, "data": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取技术详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.get("/dev-api/api/attck/technique/detail")
async def get_attck_technique_detail(
    file_name: str = Query(...),
    alias: str = Query(...),
    current_user: dict = Depends(get_current_user)
):
    """获取技术详情（通过文件名和别名）"""
    result = query_attck_technique_detail(file_name, alias)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', '未找到'))
    
    return result


@router.get("/dev-api/api/attck/tactics-techniques")
async def get_tactics_techniques(current_user: dict = Depends(get_current_user)):
    """获取所有战术和技术"""
    conn = None
    cursor = None
    try:
        conn = get_attck_connection()
        cursor = conn.cursor()
        
        sql = """
        SELECT DISTINCT tactic_id, tactic_name, technique_id, technique_name
        FROM tactics_technique
        WHERE technique_id IS NOT NULL
        ORDER BY tactic_id, technique_id
        """
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        return {"success": True, "data": results}
        
    except Exception as e:
        logger.error(f"获取战术技术失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.get("/dev-api/api/attck/api-components")
async def get_api_components(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: dict = Depends(get_current_user)
):
    """获取API组件映射列表"""
    conn = None
    cursor = None
    try:
        conn = get_attck_connection()
        cursor = conn.cursor()

        # 基础查询
        base_sql = """
            SELECT
                mi.file_name as hash_id,
                mi.alias as api_component,
                mi.root_function,
                mi.created_at,
                GROUP_CONCAT(DISTINCT ti.technique_id) as technique_ids
            FROM main_info mi
            LEFT JOIN technique_info ti ON mi.file_name = ti.file_name AND mi.alias = ti.alias
        """

        # 条件和参数
        conditions = []
        params = []

        if search:
            conditions.append("""
                (mi.file_name LIKE %s OR mi.alias LIKE %s OR
                 mi.root_function LIKE %s OR ti.technique_id LIKE %s)
            """)
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param, search_param])

        # 拼接WHERE子句
        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)

        # 分组和排序
        base_sql += " GROUP BY mi.file_name, mi.alias, mi.root_function, mi.created_at"
        base_sql += " ORDER BY mi.created_at DESC"

        # 总条数查询
        count_sql = """
            SELECT COUNT(DISTINCT CONCAT(mi.file_name, '|', mi.alias)) as total
            FROM main_info mi
            LEFT JOIN technique_info ti ON mi.file_name = ti.file_name AND mi.alias = ti.alias
        """
        if conditions:
            count_sql += " WHERE " + " AND ".join(conditions)

        cursor.execute(count_sql, params)
        total = cursor.fetchone()['total']

        # 分页查询
        offset = (page - 1) * page_size
        final_sql = f"{base_sql} LIMIT %s OFFSET %s"
        params.extend([page_size, offset])

        cursor.execute(final_sql, params)
        results = cursor.fetchall()

        # 处理technique_ids，转换为列表
        result_list = []
        start_index = (page - 1) * page_size  # 计算当前页的起始序号

        for index, item in enumerate(results):
            technique_ids_str = item.get('technique_ids', '')
            if technique_ids_str:
                # 分割technique_id，去重并排序
                technique_ids = list(set([tid.strip() for tid in technique_ids_str.split(',') if tid.strip()]))
                technique_ids.sort()  # 按字母顺序排序
            else:
                technique_ids = []

            # 构建返回数据，使用顺序编号作为id
            component_item = {
                "id": start_index + index + 1,  # 使用顺序编号作为ID
                "hash_id": item.get('hash_id', ''),
                "api_component": item.get('api_component', ''),
                "root_function": item.get('root_function', ''),
                "technique_ids": technique_ids,
                "created_at": item.get('created_at').isoformat() if item.get('created_at') else None
            }
            result_list.append(component_item)

        return {
            "success": True,
            "data": result_list,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size
            }
        }

    except Exception as e:
        logger.error(f"获取API组件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.get("/dev-api/api/attck/api-component/detail")
async def get_api_component_detail(
    hash_id: Optional[str] = Query(None, description="文件名(SHA256)"),
    api_component: Optional[str] = Query(None, description="API组件别名"),
    current_user: dict = Depends(get_current_user)
):
    """获取API组件详情"""
    if not hash_id or not api_component:
        raise HTTPException(status_code=400, detail="hash_id和api_component参数不能为空")

    # 使用query_attck_technique_detail函数,因为逻辑是一样的
    result = query_attck_technique_detail(hash_id, api_component)

    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', '未找到'))

    return result


@router.get("/dev-api/api/attck/technique-mapping")
async def get_technique_mapping(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="搜索关键词"),
    current_user: dict = Depends(get_current_user)
):
    """获取技术映射列表"""
    conn = None
    cursor = None
    try:
        conn = get_attck_connection()
        cursor = conn.cursor()

        # 基础查询：按技术编号分组，统计数量，并关联战术名称
        base_sql = """
            SELECT
                ti.technique_id,
                MAX(ti.technique_name) as technique_name,
                MAX(tt.tactic_name) as tactic_name,
                COUNT(DISTINCT CONCAT(ti.file_name, '|', ti.alias)) as function_count
            FROM technique_info ti
            LEFT JOIN tactics_technique tt ON ti.technique_id = tt.technique_id
        """

        # 条件和参数
        conditions = []
        params = []

        if search:
            conditions.append("""
                (ti.technique_id LIKE %s OR
                 ti.technique_name LIKE %s OR
                 tt.tactic_name LIKE %s)
            """)
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param])

        # 拼接WHERE子句
        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)

        # 分组和排序
        base_sql += " GROUP BY ti.technique_id"
        base_sql += " ORDER BY ti.technique_id"

        # 总条数查询
        count_sql = f"SELECT COUNT(DISTINCT technique_id) as total FROM ({base_sql}) as temp"
        cursor.execute(count_sql, params)
        total = cursor.fetchone()['total']

        # 分页处理
        offset = (page - 1) * page_size
        final_sql = f"{base_sql} LIMIT %s OFFSET %s"
        params.extend([page_size, offset])

        cursor.execute(final_sql, params)
        results = cursor.fetchall()

        # 处理结果，添加序号
        result_list = []
        start_index = (page - 1) * page_size

        for index, item in enumerate(results):
            mapping_item = {
                "id": start_index + index + 1,
                "technique_id": item.get('technique_id', ''),
                "technique_name": item.get('technique_name', ''),
                "tactic_name": item.get('tactic_name', ''),
                "function_count": item.get('function_count', 0)
            }
            result_list.append(mapping_item)

        return {
            "success": True,
            "data": result_list,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size
            }
        }
        
    except Exception as e:
        logger.error(f"获取技术映射失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.get("/dev-api/api/attck/technique-functions")
async def get_technique_functions(
    technique_id: str = Query(..., description="技术编号"),
    current_user: dict = Depends(get_current_user)
):
    """获取指定技术编号对应的函数列表"""
    conn = None
    cursor = None
    try:
        conn = get_attck_connection()
        cursor = conn.cursor()

        # 查询指定技术编号对应的所有函数
        sql = """
            SELECT
                alias as function_name,
                file_name
            FROM technique_info
            WHERE technique_id = %s
            ORDER BY alias, file_name
        """

        cursor.execute(sql, [technique_id])
        results = cursor.fetchall()

        # 处理结果
        function_list = []
        for index, item in enumerate(results):
            function_item = {
                "id": index + 1,
                "function_name": item.get('function_name', ''),
                "file_name": item.get('file_name', '')
            }
            function_list.append(function_item)

        return {
            "success": True,
            "data": function_list,
            "technique_id": technique_id,
            "total": len(function_list)
        }
        
    except Exception as e:
        logger.error(f"获取技术函数失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
