"""
分布式杀毒软件扫描API
功能: 单个文件检测、批量文件检测、任务进度查询、CSV报告下载

架构 v2: 本模块为薄代理层，所有 AV 扫描请求转发到独立检测服务 (:5006)。
         认证在此层处理，排队和 VM 通信由独立服务负责。
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
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
import asyncio
import requests

# 批量任务本地队列（仅用于本地追踪，实际排队由独立服务处理）
from app.services.av_detection.scan_queue import join_queue, leave_queue

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================
# 配置：独立检测服务地址
# ============================================================
AV_SERVICE_URL = getattr(settings, "AV_SERVICE_URL", os.environ.get("AV_SERVICE_URL", "http://127.0.0.1:5006"))

# 降级模式：如果独立服务不可用，回退到直接客户端
_use_direct_client = False
_av_client = None

if os.environ.get("AV_USE_DIRECT_CLIENT", "").lower() == "true":
    _use_direct_client = True
    try:
        from app.services.av_detection import AVDistributedClient
        config_path = Path(__file__).parent.parent / "services" / "av_detection" / "vm_config.json"
        _av_client = AVDistributedClient(config_path=str(config_path))
        logger.info(f"降级模式: 使用直接 AV 客户端, config={config_path}")
    except Exception as e:
        logger.error(f"直接客户端初始化失败: {e}")

# 保留旧引用兼容
av_client = _av_client

# 批量任务存储(生产环境应使用Redis)
batch_tasks: Dict[str, Dict[str, Any]] = {}
BATCH_TASK_ID_RE = re.compile(r"^batch_\d{8}_\d{6}_[0-9a-f]{8}$")
BATCH_TASKS_ROOT = (Path(settings.UPLOAD_DIR) / "batch_tasks").resolve()

# 杀软引擎列表
AV_ENGINES = [
    "Avira", "McAfee", "WindowsDefender", "IkarusT3", "Emsisoft",
    "FProtect", "Vba32", "ClamAV", "Kaspersky", "ESET",
    "DrWeb", "Avast", "AVG", "AdAware", "FSecure"
]


def _is_valid_batch_task_id(task_id: str) -> bool:
    """校验批量任务ID格式。"""
    return bool(BATCH_TASK_ID_RE.fullmatch(task_id or ""))


def _safe_resolve_path(root: Path, *parts: str) -> Path:
    """在固定根目录下解析路径，防止目录穿越。"""
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts).resolve()
    candidate.relative_to(root_resolved)
    return candidate


@router.post("/av_scan_single")
async def scan_single_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    单个文件杀毒软件检测（代理到独立检测服务 :5006）
    """
    if _use_direct_client:
        return await _scan_single_direct(file)

    try:
        content = await file.read()

        def _call():
            resp = requests.post(
                f"{AV_SERVICE_URL}/api/av_scan_single",
                files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json()

        return await asyncio.to_thread(_call)

    except requests.HTTPError as e:
        logger.error(f"检测服务返回错误: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except requests.RequestException as e:
        logger.error(f"无法连接到独立检测服务 {AV_SERVICE_URL}: {e}")
        raise HTTPException(status_code=503, detail=f"检测服务不可用: {AV_SERVICE_URL}")
    except Exception as e:
        logger.error(f"扫描失败: {e}")
        raise HTTPException(status_code=500, detail=f"扫描失败: {e}")


async def _scan_single_direct(file: UploadFile):
    """降级：直接使用本地 AV 客户端"""
    from app.services.av_detection.scan_queue import SCAN_LOCK, join_queue, leave_queue, get_position, set_current, clear_current

    if _av_client is None:
        raise HTTPException(status_code=500, detail="AV客户端未初始化")

    task_id = uuid.uuid4().hex[:8]
    join_queue(task_id)
    try:
        async with SCAN_LOCK:
            set_current({"type": "single_scan", "task_id": task_id, "file_name": file.filename})
            upload_dir = Path(settings.UPLOAD_DIR) / "av_scan_temp"
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / file.filename
            content = await file.read()
            file_path.write_bytes(content)
            file_size_str = f"{len(content) / (1024*1024):.2f} MB"

            scan_result = await asyncio.to_thread(_av_client.scan_single_file, str(file_path))
            formatted = format_single_scan_result(scan_result, file.filename, file_size_str)

            if file_path.exists():
                os.remove(file_path)
            clear_current()
            return formatted
    finally:
        leave_queue(task_id)
@router.post("/av_scan_single_streaming")
async def scan_single_file_streaming(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    单个文件流式扫描（代理 SSE 流从独立检测服务 :5006）
    """
    if _use_direct_client:
        return await _scan_single_streaming_direct(file)

    raw_content = await file.read()

    async def generate():
        import threading
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()

        def _stream():
            try:
                resp = requests.post(
                    f"{AV_SERVICE_URL}/api/av_scan_single_streaming",
                    files={"file": (file.filename, raw_content, file.content_type or "application/octet-stream")},
                    stream=True,
                    timeout=600,
                )
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        loop.call_soon_threadsafe(
                            q.put_nowait,
                            line.decode("utf-8", errors="replace") + "\n",
                        )
            except Exception as e:
                loop.call_soon_threadsafe(
                    q.put_nowait,
                    f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n",
                )
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=_stream, daemon=True).start()

        while True:
            chunk = await q.get()
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _scan_single_streaming_direct(file: UploadFile):
    """降级：直接使用本地 AV 客户端流式扫描"""
    from app.services.av_detection.scan_queue import SCAN_LOCK, join_queue, leave_queue, get_position, set_current, clear_current

    if _av_client is None:
        raise HTTPException(status_code=500, detail="AV客户端未初始化")

    raw_content = await file.read()
    file_size_str = f"{len(raw_content) / (1024*1024):.2f} MB"
    task_id = uuid.uuid4().hex[:8]
    join_queue(task_id)

    async def generate():
        try:
            while True:
                pos = get_position(task_id)
                if pos <= 0:
                    yield f"data: {json.dumps({'type': 'error', 'error': '任务已被取消'})}\n\n"
                    return
                if pos == 1:
                    break
                yield f"data: {json.dumps({'type': 'queued', 'position': pos, 'message': f'排队中，前面还有 {pos - 1} 个任务'})}\n\n"
                await asyncio.sleep(1)

            async with SCAN_LOCK:
                set_current({"type": "single_scan_streaming", "task_id": task_id, "file_name": file.filename})
                upload_dir = Path(settings.UPLOAD_DIR) / "av_scan_temp"
                upload_dir.mkdir(parents=True, exist_ok=True)
                file_path = upload_dir / file.filename
                file_path.write_bytes(raw_content)

                yield f"data: {json.dumps({'type': 'file_info', 'file_name': file.filename, 'file_size': file_size_str})}\n\n"

                results = await asyncio.to_thread(lambda: list(_av_client.scan_single_file_streaming(
                    file_content=raw_content, file_name=file.filename,
                )))
                for r in results:
                    yield f"data: {json.dumps(r)}\n\n"

                if file_path.exists():
                    os.remove(file_path)
                clear_current()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            leave_queue(task_id)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


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

            # 获取检测标签
            label = file_result.get('labels', {}).get(engine_name, '')

            # 从engine_details中获取耗时
            elapsed = 0
            for engine_detail in scan_result.get('engine_details', []):
                if engine_detail.get('engine') == engine_name:
                    elapsed = engine_detail.get('elapsed_seconds', 0)
                    break

            engines_result.append({
                "name": engine_name,
                "status": status,
                "label": label,
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
        task_dir = _safe_resolve_path(BATCH_TASKS_ROOT, task_id)
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
            "user_id": current_user.get("id", 0),  # 新增：用户ID
            "task_dir": str(task_dir)
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
    启动批量检测任务（代理到独立检测服务）
    """
    task_id = request.get('task_id')
    if not task_id:
        raise HTTPException(status_code=400, detail="缺少task_id参数")

    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = batch_tasks[task_id]
    if task['status'] != 'pending':
        raise HTTPException(status_code=400, detail=f"任务状态不正确: {task['status']}")

    # 加入 :5005 本地排队
    join_queue(task_id)

    # 注册到 :5006 内部队列（统一排队，所有任务可见）
    remote_pos = 1
    try:
        def _register_remote():
            resp = requests.post(f"{AV_SERVICE_URL}/api/av_scan_queue_register",
                json={"task_id": task_id, "type": "batch_scan",
                      "file_name": f"批量{len(task['files'])}个文件",
                      "file_count": len(task['files'])},
                timeout=5)
            if resp.ok:
                return resp.json().get("position", 0)
            return 1
        remote_pos = await asyncio.to_thread(_register_remote)
        logger.info(f"[批次 {task_id}] 在 :5006 队列位置: {remote_pos}")
    except requests.RequestException:
        pass

    task['status'] = 'queued' if remote_pos > 1 else 'pending'
    task['start_time'] = datetime.now()

    # 在后台执行批量扫描
    background_tasks.add_task(execute_batch_scan, task_id)

    logger.info(f"批量检测任务已启动: task_id={task_id}  status={task['status']}")

    return {
        "task_id": task_id,
        "status": task['status'],
        "message": "批量检测任务已启动" + ("（排队中）" if task['status'] == 'queued' else "")
    }


async def execute_batch_scan(task_id: str):
    """执行批量扫描任务(通过独立检测服务)"""
    from app.services.av_detection.scan_queue import SCAN_LOCK, set_current, clear_current

    try:
        task = batch_tasks[task_id]
        files = task['files']
        total_files = len(files)
        selected_engines = task.get('selected_engines', AV_ENGINES)

        # —— 在 :5006 队列中等待，直到排到第 1 ——
        task['status'] = 'queued'
        logger.info(f"[批次 {task_id}] 在 :5006 队列中等待...")
        while True:
            try:
                def _check_pos():
                    resp = requests.get(f"{AV_SERVICE_URL}/api/av_scan_queue_position/{task_id}", timeout=5)
                    if resp.ok:
                        return resp.json().get("position", -1)
                    return -1
                pos = await asyncio.to_thread(_check_pos)
            except requests.RequestException:
                pos = -1

            if pos <= 0:
                # 不在队列中了（可能被取消）
                task['status'] = 'failed'
                task['error'] = '任务已从队列中移除'
                return
            if pos == 1:
                break
            await asyncio.sleep(2)

        # —— 排到了，离开 :5006 队列，开始扫描 ——
        try:
            def _leave_remote():
                requests.post(f"{AV_SERVICE_URL}/api/av_scan_queue_unregister",
                    json={"task_id": task_id}, timeout=5)
            await asyncio.to_thread(_leave_remote)
        except requests.RequestException:
            pass

        async with SCAN_LOCK:
            set_current({"type": "batch_scan", "task_id": task_id, "files": total_files, "engines": selected_engines})
            task['status'] = 'running'
            task['start_time'] = datetime.now()
            logger.info(f"[批次 {task_id}] 开始扫描 {total_files} 个文件")

            async def scan_one(file_info, idx):
                try:
                    task['current_file'] = file_info['name']
                    file_path = file_info['path']

                    def _call():
                        with open(file_path, "rb") as f:
                            file_content = f.read()
                        resp = requests.post(
                            f"{AV_SERVICE_URL}/api/av_scan_single",
                            files={"file": (file_info['name'], file_content, "application/octet-stream")},
                            data={"engines": ",".join(selected_engines)},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        return resp.json()

                    scan_result = await asyncio.to_thread(_call)

                    formatted = format_batch_scan_result_from_service(scan_result, file_info['name'])
                    task['scanned_files'] = idx + 1
                    task['progress'] = (idx + 1) / total_files * 100
                    logger.info(f"[进度] scanned_files={idx+1}/{total_files}, progress={task['progress']:.1f}%")
                    return formatted

                except Exception as e:
                    logger.error(f"扫描文件失败: {file_info['name']}, error={e}")
                    task['scanned_files'] = idx + 1
                    task['progress'] = (idx + 1) / total_files * 100
                    return {"file_name": file_info['name'], "error": str(e), "engines": {}}

            # 改为逐个扫描（不并发），实时更新进度
            results = []
            for i, file_info in enumerate(files):
                r = await scan_one(file_info, i)
                results.append(r)

        task['results'] = results
        task['status'] = 'completed'
        task['end_time'] = datetime.now()
        task['current_file'] = None
        logger.info(f"批量扫描完成: task_id={task_id}")

        # 保存历史记录
        try:
            from app.api.av_scan_history import save_scan_to_history
            save_scan_to_history(
                task_id=task_id,
                user_id=task.get('user_id', 0),
                status='completed',
                total_files=total_files,
                selected_engines=selected_engines,
                scan_results=task['results'],
            )
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")

    except requests.RequestException as e:
        logger.error(f"无法连接检测服务 {AV_SERVICE_URL}: {e}")
        task['status'] = 'failed'
        task['error'] = f"检测服务不可用: {AV_SERVICE_URL}"
    except Exception as e:
        logger.error(f"批量扫描任务异常: task_id={task_id}, error={e}")
        task['status'] = 'failed'
        task['error'] = str(e)
    finally:
        clear_current()
        leave_queue(task_id)
        # 从 :5006 的统一队列视图注销
        try:
            def _unregister_remote():
                requests.post(f"{AV_SERVICE_URL}/api/av_scan_queue_unregister",
                    json={"task_id": task_id}, timeout=5)
            await asyncio.to_thread(_unregister_remote)
        except Exception:
            pass


def format_batch_scan_result_from_service(scan_result: Dict, file_name: str) -> Dict:
    """将从独立服务返回的单文件扫描结果转为批量格式"""
    engines = {}; malicious_count = 0
    for engine_info in scan_result.get("engines", []):
        name = engine_info.get("name", "")
        status = engine_info.get("status", "unsupported")
        label = engine_info.get("label", "")
        engines[name] = {"status": status, "label": label}
        if status == "malicious":
            malicious_count += 1
    return {
        "file_name": file_name,
        "malicious_count": malicious_count,
        "safe_count": len(engines) - malicious_count,
        "engines": engines,
    }


def format_batch_scan_result(scan_result: Dict, file_name: str) -> Dict:
    """格式化批量扫描结果"""
    engines = {}
    malicious_count = 0

    if 'file_results' in scan_result and file_name in scan_result['file_results']:
        file_result = scan_result['file_results'][file_name]

        for engine_name, detection in file_result['engines'].items():
            label = file_result.get('labels', {}).get(engine_name, '')
            if detection == 1:
                engines[engine_name] = {"status": "malicious", "label": label}
                malicious_count += 1
            elif detection == 0:
                engines[engine_name] = {"status": "safe", "label": label}
            else:
                engines[engine_name] = {"status": "unsupported", "label": label}

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
    if not _is_valid_batch_task_id(task_id):
        raise HTTPException(status_code=400, detail="无效的任务ID格式")
    
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = batch_tasks[task_id]

    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"任务未完成,当前状态: {task['status']}")

    try:
        # 生成CSV文件
        task_dir_str = task.get('task_dir', '')
        if not task_dir_str:
            raise HTTPException(status_code=404, detail="任务目录不存在")

        task_dir = _safe_resolve_path(BATCH_TASKS_ROOT, Path(task_dir_str).name)
        csv_path = task_dir / "report.csv"
        
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
                    engine_info = result['engines'].get(engine, {})
                    if isinstance(engine_info, dict):
                        status = engine_info.get('status', 'N/A')
                    else:
                        status = engine_info or 'N/A'
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
            path=str(csv_path),
            filename=f"av_scan_report_{task_id}.csv",
            media_type='text/csv'
        )

    except Exception as e:
        logger.error(f"生成CSV报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成报告失败: {str(e)}")


@router.get("/av_engines")
async def get_av_engines(current_user: dict = Depends(get_current_user)):
    """获取可用的杀毒引擎列表（从独立服务查询）"""
    try:
        def _call():
            resp = requests.get(f"{AV_SERVICE_URL}/api/av_engines", timeout=10)
            resp.raise_for_status()
            return resp.json()
        return await asyncio.to_thread(_call)
    except requests.RequestException:
        if _use_direct_client and _av_client:
            engines = _av_client.get_available_engines()
            return {
                "total": len(engines),
                "engines": [{"name": e, "vm": _av_client.engine_to_vm.get(e, {}).get("vm_id", "unknown")} for e in engines],
            }
        raise HTTPException(status_code=503, detail=f"检测服务不可用: {AV_SERVICE_URL}")


@router.get("/av_vm_status")
async def get_av_vm_status(current_user: dict = Depends(get_current_user)):
    """获取虚拟机状态（从独立服务查询）"""
    try:
        def _call():
            resp = requests.get(f"{AV_SERVICE_URL}/api/av_vm_status", timeout=15)
            resp.raise_for_status()
            return resp.json()
        return await asyncio.to_thread(_call)
    except requests.RequestException:
        if _use_direct_client and _av_client:
            return _av_client.get_vm_status()
        raise HTTPException(status_code=503, detail=f"检测服务不可用: {AV_SERVICE_URL}")


@router.get("/av_scan_queue_status")
async def get_scan_queue_status(current_user: dict = Depends(get_current_user)):
    """查询扫描任务队列状态（合并本地批量任务 + :5006 单个扫描任务）"""
    from app.services.av_detection.scan_queue import get_status as local_get_status

    local_status = local_get_status()

    remote_status = {"running": False, "queue_length": 0, "current": None, "all_queued": []}
    try:
        def _call():
            resp = requests.get(f"{AV_SERVICE_URL}/api/av_scan_queue_status", timeout=10)
            resp.raise_for_status()
            return resp.json()
        remote_status = await asyncio.to_thread(_call)
    except requests.RequestException:
        pass  # :5006 不可达时只用本地状态

    # 合并队列（去重）
    local_queued = local_status.get("all_queued", [])
    remote_queued = remote_status.get("all_queued", [])
    # 用 dict 去重保序
    merged_queued = list(dict.fromkeys(local_queued + remote_queued))

    return {
        "running": local_status.get("running") or remote_status.get("running", False),
        "queue_length": len([x for x in merged_queued if x != (remote_status.get("current") or {}).get("task_id")]),
        "current": local_status.get("current") or remote_status.get("current"),
        "all_queued": merged_queued,
        "external_tasks": remote_status.get("external_tasks", []),
    }
