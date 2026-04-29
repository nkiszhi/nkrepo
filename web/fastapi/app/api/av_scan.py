"""
分布式杀毒软件扫描API
功能: 单个文件检测、批量文件检测、任务进度查询、CSV报告下载
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form
from fastapi.responses import FileResponse, StreamingResponse
from app.api.auth import get_current_user
from app.core import settings
from pathlib import Path
import os
import logging
import json
import csv
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)

# 导入AV分布式客户端
try:
    from app.services.av_detection import AVDistributedClient
    from pathlib import Path
    
    # 获取vm_config.json的路径
    config_path = Path(__file__).parent.parent / "services" / "av_detection" / "vm_config.json"
    
    # 初始化AV客户端
    av_client = AVDistributedClient(config_path=str(config_path))
    logger.info(f"AV分布式客户端初始化成功,配置文件: {config_path}")
except Exception as e:
    logger.error(f"AV分布式客户端初始化失败: {str(e)}")
    av_client = None

# 批量任务存储(生产环境应使用Redis)
batch_tasks: Dict[str, Dict[str, Any]] = {}

# 杀软引擎列表
AV_ENGINES = [
    "Avira", "McAfee", "WindowsDefender", "IkarusT3", "Emsisoft",
    "FProtect", "Vba32", "ClamAV", "Kaspersky", "ESET",
    "DrWeb", "Avast", "AVG", "AdAware", "FSecure"
]


@router.post("/av_scan_single")
async def scan_single_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    单个文件杀毒软件检测
    使用15个杀毒引擎并行检测单个文件
    """
    if av_client is None:
        raise HTTPException(status_code=500, detail="AV分布式客户端未初始化")

    try:
        # 保存上传文件
        upload_dir = Path(settings.UPLOAD_DIR) / "av_scan_temp"
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 获取文件大小
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_size_str = f"{file_size_mb:.2f} MB"

        logger.info(f"开始扫描文件: {file.filename}, 大小: {file_size_str}")

        # 调用AV分布式客户端进行扫描
        scan_result = av_client.scan_single_file(str(file_path))

        # 格式化结果
        formatted_result = format_single_scan_result(scan_result, file.filename, file_size_str)

        # 清理临时文件
        if file_path.exists():
            os.remove(file_path)

        logger.info(f"文件扫描完成: {file.filename}")
        return formatted_result

    except Exception as e:
        logger.error(f"文件扫描失败: {str(e)}")
        # 清理临时文件
        if 'file_path' in locals() and file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"扫描失败: {str(e)}")


@router.post("/av_scan_single_streaming")
async def scan_single_file_streaming(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    单个文件杀毒软件检测 - 流式返回结果
    每完成一个引擎就返回结果,实现实时显示
    """
    if av_client is None:
        raise HTTPException(status_code=500, detail="AV分布式客户端未初始化")

    try:
        # 保存上传文件
        upload_dir = Path(settings.UPLOAD_DIR) / "av_scan_temp"
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 获取文件大小
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_size_str = f"{file_size_mb:.2f} MB"

        logger.info(f"开始流式扫描文件: {file.filename}, 大小: {file_size_str}")

        # 生成器函数,用于流式返回结果
        async def generate():
            try:
                # 先发送文件信息
                file_info = {
                    'type': 'file_info',
                    'file_name': file.filename,
                    'file_size': file_size_str
                }
                yield f"data: {json.dumps(file_info)}\n\n"

                # 调用流式扫描
                for result in av_client.scan_single_file_streaming(str(file_path)):
                    yield f"data: {json.dumps(result)}\n\n"

                logger.info(f"流式扫描完成: {file.filename}")

            finally:
                # 清理临时文件
                if file_path.exists():
                    os.remove(file_path)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"流式扫描失败: {str(e)}")
        # 清理临时文件
        if 'file_path' in locals() and file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"扫描失败: {str(e)}")


def format_single_scan_result(scan_result: Dict, file_name: str, file_size: str) -> Dict:
    """格式化单个文件扫描结果"""
    engines_result = []
    malicious_count = 0
    safe_count = 0
    unsupported_count = 0
    error_count = 0

    # 从file_results中提取结果
    if 'file_results' in scan_result and file_name in scan_result['file_results']:
        file_result = scan_result['file_results'][file_name]

        for engine_name, detection in file_result['engines'].items():
            if detection == 1:
                status = "malicious"
                malicious_count += 1
            elif detection == 0:
                status = "safe"
                safe_count += 1
            else:
                status = "unsupported"
                unsupported_count += 1

            # 从engine_details中获取耗时
            elapsed = 0
            for engine_detail in scan_result.get('engine_details', []):
                if engine_detail.get('engine') == engine_name:
                    elapsed = engine_detail.get('elapsed_seconds', 0)
                    break

            engines_result.append({
                "name": engine_name,
                "status": status,
                "vm": get_engine_vm(engine_name),
                "elapsed_seconds": round(elapsed, 3)
            })

    # 处理失败的引擎
    for engine_detail in scan_result.get('engine_details', []):
        if not engine_detail.get('success'):
            engines_result.append({
                "name": engine_detail.get('engine'),
                "status": "error",
                "vm": engine_detail.get('vm_id', 'unknown'),
                "elapsed_seconds": 0,
                "error": engine_detail.get('error', '未知错误')
            })
            error_count += 1

    return {
        "file_name": file_name,
        "file_size": file_size,
        "scan_time": scan_result.get('scan_time', datetime.now().isoformat()),
        "elapsed_seconds": scan_result.get('elapsed_seconds', 0),
        "total_engines": scan_result.get('total_engines', 15),
        "malicious_count": malicious_count,
        "safe_count": safe_count,
        "unsupported_count": unsupported_count,
        "error_count": error_count,
        "engines": engines_result
    }


def get_engine_vm(engine_name: str) -> str:
    """获取引擎所在的虚拟机"""
    if av_client and engine_name in av_client.engine_to_vm:
        return av_client.engine_to_vm[engine_name]['vm_id']
    return "unknown"


@router.post("/av_batch_upload")
async def batch_upload_files(
    files: List[UploadFile] = File(...),
    engines: str = Form(""),  # 修复：使用 Form() 接收 FormData 中的字符串参数
    current_user: dict = Depends(get_current_user)
):
    """
    批量文件上传
    创建批量任务ID,接收多个文件上传
    支持指定引擎列表
    """
    try:
        # 生成任务ID
        task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 创建任务目录
        task_dir = Path(settings.UPLOAD_DIR) / "batch_tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # 保存上传的文件
        uploaded_files = []
        for file in files:
            file_path = task_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)

            uploaded_files.append({
                "name": file.filename,
                "size": f"{file_size_mb:.2f} MB",
                "path": str(file_path)
            })

        # 解析引擎列表
        selected_engines = []
        logger.info(f"接收到的engines参数: '{engines}', 类型: {type(engines)}")
        if engines:
            selected_engines = [e.strip() for e in engines.split(",") if e.strip()]
            logger.info(f"解析后的引擎列表: {selected_engines}")
        else:
            # 默认使用所有引擎
            selected_engines = AV_ENGINES
            logger.info(f"未指定引擎，使用默认所有引擎: {len(selected_engines)}个")

        # 初始化任务状态
        batch_tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "total_files": len(files),
            "scanned_files": 0,
            "current_file": None,
            "start_time": datetime.now(),
            "files": uploaded_files,
            "results": [],
            "error": None,
            "selected_engines": selected_engines,  # 新增：选择的引擎
            "user_id": current_user.get("id", 0)  # 新增：用户ID
        }

        logger.info(f"批量上传完成: task_id={task_id}, files={len(files)}, engines={len(selected_engines)}")

        return {
            "task_id": task_id,
            "upload_dir": str(task_dir),
            "files": [{"name": f["name"], "size": f["size"]} for f in uploaded_files],
            "total_files": len(files),
            "selected_engines": selected_engines  # 新增：返回选择的引擎
        }

    except Exception as e:
        logger.error(f"批量上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/av_batch_scan_start")
async def start_batch_scan(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    启动批量检测任务
    在后台异步执行批量扫描
    """
    if av_client is None:
        raise HTTPException(status_code=500, detail="AV分布式客户端未初始化")

    task_id = request.get('task_id')
    if not task_id:
        raise HTTPException(status_code=400, detail="缺少task_id参数")

    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = batch_tasks[task_id]
    if task['status'] != 'pending':
        raise HTTPException(status_code=400, detail=f"任务状态不正确: {task['status']}")

    # 更新任务状态
    task['status'] = 'running'
    task['start_time'] = datetime.now()

    # 在后台执行批量扫描
    background_tasks.add_task(execute_batch_scan, task_id)

    logger.info(f"批量检测任务已启动: task_id={task_id}")

    return {
        "task_id": task_id,
        "status": "running",
        "message": "批量检测任务已启动"
    }


async def execute_batch_scan(task_id: str):
    """执行批量扫描任务(后台任务) - 优化版：文件级并行扫描"""
    try:
        task = batch_tasks[task_id]
        files = task['files']
        total_files = len(files)
        selected_engines = task.get('selected_engines', AV_ENGINES)  # 获取选择的引擎

        logger.info(f"开始执行批量扫描: task_id={task_id}, files={total_files}, engines={len(selected_engines)}")

        # 定义单个文件扫描函数
        async def scan_single_file_task(file_info: dict, idx: int) -> dict:
            """扫描单个文件的异步任务"""
            try:
                file_path = file_info['path']
                file_name = file_info['name']

                logger.info(f"扫描文件 [{idx+1}/{total_files}]: {file_name}")

                # 更新当前正在扫描的文件
                task['current_file'] = file_name

                # 调用AV客户端扫描 - 使用选择的引擎
                # 优化：max_workers设为引擎数量，timeout减少到30秒
                scan_result = await asyncio.to_thread(
                    av_client.scan_single_file,
                    file_path,
                    selected_engines,  # 传递选择的引擎列表
                    len(selected_engines),  # max_workers: 动态设置为引擎数量
                    60   # timeout: 
                )

                # 格式化结果
                formatted = format_batch_scan_result(scan_result, file_name)

                # 更新进度
                task['scanned_files'] = idx + 1
                task['progress'] = (idx + 1) / total_files * 100
                logger.info(f"[进度更新] scanned_files={idx+1}, progress={task['progress']:.1f}%")

                return formatted

            except Exception as e:
                logger.error(f"扫描文件失败: {file_info['name']}, error={str(e)}")
                # 返回错误结果
                task['scanned_files'] = idx + 1
                task['progress'] = (idx + 1) / total_files * 100
                return {
                    "file_name": file_info['name'],
                    "error": str(e),
                    "engines": {}
                }

        # 使用 asyncio.gather 并行扫描所有文件
        # 限制并发数为 min(文件数, 5) 避免资源耗尽
        max_concurrent = min(total_files, 5)
        results = []
        
        # 分批处理，每批 max_concurrent 个文件
        for batch_start in range(0, total_files, max_concurrent):
            batch_end = min(batch_start + max_concurrent, total_files)
            batch_files = files[batch_start:batch_end]
            
            # 并行扫描当前批次
            batch_tasks_list = [
                scan_single_file_task(file_info, batch_start + idx) 
                for idx, file_info in enumerate(batch_files)
            ]
            batch_results = await asyncio.gather(*batch_tasks_list)
            results.extend(batch_results)
            
            logger.info(f"批次完成: {batch_start}-{batch_end}/{total_files}")

        # 保存结果
        task['results'] = results

        # 任务完成
        task['status'] = 'completed'
        task['end_time'] = datetime.now()
        task['current_file'] = None  # 清空当前文件
        logger.info(f"批量扫描完成: task_id={task_id}")

        # 保存到历史记录
        try:
            from app.api.av_scan_history import save_scan_to_history
            save_scan_to_history(
                task_id=task_id,
                user_id=task.get('user_id', 0),
                status='completed',
                total_files=total_files,
                selected_engines=selected_engines,
                scan_results=task['results']
            )
        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}")

    except Exception as e:
        logger.error(f"批量扫描任务异常: task_id={task_id}, error={str(e)}")
        task['status'] = 'failed'
        task['error'] = str(e)


def format_batch_scan_result(scan_result: Dict, file_name: str) -> Dict:
    """格式化批量扫描结果"""
    engines = {}
    malicious_count = 0

    if 'file_results' in scan_result and file_name in scan_result['file_results']:
        file_result = scan_result['file_results'][file_name]

        for engine_name, detection in file_result['engines'].items():
            if detection == 1:
                engines[engine_name] = "malicious"
                malicious_count += 1
            elif detection == 0:
                engines[engine_name] = "safe"
            else:
                engines[engine_name] = "unsupported"

    return {
        "file_name": file_name,
        "malicious_count": malicious_count,
        "safe_count": len(engines) - malicious_count,
        "engines": engines
    }


@router.get("/av_batch_scan_status/{task_id}")
async def get_batch_scan_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    查询批量检测任务进度
    """
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = batch_tasks[task_id]

    # 计算已用时间和预计剩余时间
    elapsed = (datetime.now() - task['start_time']).total_seconds()
    estimated_remaining = 0

    if task['scanned_files'] > 0 and task['scanned_files'] < task['total_files']:
        avg_time_per_file = elapsed / task['scanned_files']
        remaining_files = task['total_files'] - task['scanned_files']
        estimated_remaining = avg_time_per_file * remaining_files

    return {
        "task_id": task_id,
        "status": task['status'],
        "progress": round(task['progress'], 2),
        "total_files": task['total_files'],
        "scanned_files": task['scanned_files'],
        "current_file": task.get('current_file'),
        "elapsed_seconds": round(elapsed, 2),
        "estimated_remaining": round(estimated_remaining, 2),
        "error": task.get('error')
    }


@router.get("/av_batch_scan_result/{task_id}")
async def get_batch_scan_result(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    查询批量检测结果
    """
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = batch_tasks[task_id]

    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"任务未完成,当前状态: {task['status']}")

    return {
        "task_id": task_id,
        "status": task['status'],
        "scan_time": task['start_time'].isoformat(),
        "total_files": task['total_files'],
        "total_engines": 15,
        "results": task['results']
    }


@router.get("/av_batch_scan_download/{task_id}")
async def download_batch_scan_report(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    下载批量检测CSV报告
    """
    # 验证task_id格式，防止路径遍历攻击
    import re
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", task_id):
        raise HTTPException(status_code=400, detail="无效的任务ID格式")
    
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = batch_tasks[task_id]

    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"任务未完成,当前状态: {task['status']}")

    try:
        # 生成CSV文件
        csv_path = Path(settings.UPLOAD_DIR) / "batch_tasks" / task_id / "report.csv"
        
        # 获取选择的引擎列表
        selected_engines = task.get('selected_engines', AV_ENGINES)

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)

            # 写入表头 - 只包含选择的引擎
            header = ['文件名'] + selected_engines
            writer.writerow(header)

            # 写入数据 - 只包含选择的引擎
            for result in task['results']:
                row = [result['file_name']]
                for engine in selected_engines:
                    status = result['engines'].get(engine, 'N/A')
                    # 转换状态为中文
                    if status == 'malicious':
                        status_cn = '恶意'
                    elif status == 'safe':
                        status_cn = '安全'
                    elif status == 'unsupported':
                        status_cn = '不支持'
                    else:
                        status_cn = 'N/A'
                    row.append(status_cn)
                writer.writerow(row)

        logger.info(f"CSV报告生成成功: {csv_path}")

        # 返回文件下载
        return FileResponse(
            path=csv_path,
            filename=f"av_scan_report_{task_id}.csv",
            media_type='text/csv'
        )

    except Exception as e:
        logger.error(f"生成CSV报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成报告失败: {str(e)}")


@router.get("/av_engines")
async def get_av_engines(current_user: dict = Depends(get_current_user)):
    """
    获取可用的杀毒引擎列表
    """
    if av_client is None:
        raise HTTPException(status_code=500, detail="AV分布式客户端未初始化")

    try:
        engines = av_client.get_available_engines()
        engine_list = []

        for engine in engines:
            vm_info = av_client.engine_to_vm.get(engine, {})
            engine_list.append({
                "name": engine,
                "vm": vm_info.get('vm_id', 'unknown')
            })

        return {
            "total": len(engine_list),
            "engines": engine_list
        }

    except Exception as e:
        logger.error(f"获取引擎列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取引擎列表失败: {str(e)}")


@router.get("/av_vm_status")
async def get_av_vm_status(current_user: dict = Depends(get_current_user)):
    """
    获取虚拟机状态
    """
    if av_client is None:
        raise HTTPException(status_code=500, detail="AV分布式客户端未初始化")

    try:
        vm_status = av_client.get_vm_status()
        return vm_status

    except Exception as e:
        logger.error(f"获取虚拟机状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取虚拟机状态失败: {str(e)}")
