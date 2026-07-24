import sys
import os
import json
import uuid
import csv
import re
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import uvicorn

# ---- 路径设置：确保能找到依赖模块 ----
HERE = Path(__file__).resolve().parent

# 尝试从当前仓库的 Web 项目导入 AV 客户端模块
WEB_SERVICES = HERE.parent / "web" / "fastapi" / "app" / "services" / "av_detection"
if str(WEB_SERVICES) not in sys.path:
    sys.path.insert(0, str(WEB_SERVICES))

# 也可以从同目录导入
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from AV_Distributed_Client import AVDistributedClient
except ImportError:
    # 降级：尝试从本地副本导入
    try:
        from av_service.AV_Distributed_Client import AVDistributedClient
    except ImportError:
        raise ImportError(
            "Cannot import AVDistributedClient. "
            "Copy it to: " + str(HERE / "AV_Distributed_Client.py")
        )

try:
    from scan_queue import (
        SCAN_LOCK, join_queue, leave_queue, get_position,
        set_current, clear_current, get_status,
    )
except ImportError:
    try:
        from av_service.scan_queue import (
            SCAN_LOCK, join_queue, leave_queue, get_position,
            set_current, clear_current, get_status,
        )
    except ImportError:
        raise ImportError(
            "Cannot import scan_queue. "
            "Copy it to: " + str(HERE / "scan_queue.py")
        )

# ---- 配置 ------------------------------------------------------------

SERVICE_PORT = int(os.environ.get("AV_SERVICE_PORT", "5006"))
UPLOAD_DIR = Path(r"C:\av_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "av_scan_temp").mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "probe_temp").mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "batch_tasks").mkdir(parents=True, exist_ok=True)

# VM 配置 JSON（优先用同目录的，否则用 Web 项目的）
VM_CONFIG_CANDIDATES = [
    HERE / "vm_config.json",
    HERE.parent / "共享" / "vm_config.json",
    WEB_SERVICES / "vm_config.json",
]
VM_CONFIG = None
for candidate in VM_CONFIG_CANDIDATES:
    if candidate.exists():
        VM_CONFIG = str(candidate)
        break
if not VM_CONFIG:
    raise FileNotFoundError("vm_config.json not found. Searched: " + ", ".join(str(c) for c in VM_CONFIG_CANDIDATES))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("av_detector")

# ---- FastAPI 应用 ----------------------------------------------------

app = FastAPI(
    title="AV Detector Service",
    description="独立杀毒检测服务 — 提供 15 引擎分布式扫描 / 边界探测",
    version="2.0.0",
)

router = APIRouter()

# ---- AV 客户端初始化 ------------------------------------------------

av_client: AVDistributedClient = None

# 批量任务存储
batch_tasks: Dict[str, Dict[str, Any]] = {}
BATCH_TASK_ID_RE = re.compile(r"^batch_\d{8}_\d{6}_[0-9a-f]{8}$")
BATCH_TASKS_ROOT = (UPLOAD_DIR / "batch_tasks").resolve()

# 引擎列表
AV_ENGINES = [
    "Avira", "McAfee", "WindowsDefender", "IkarusT3", "Emsisoft",
    "FProtect", "Vba32", "ClamAV", "Kaspersky", "ESET",
    "DrWeb", "Avast", "AVG", "AdAware", "FSecure",
]


def get_engine_vm(engine_name: str) -> str:
    """获取引擎所在的 VM ID"""
    if av_client and engine_name in av_client.engine_to_vm:
        return av_client.engine_to_vm[engine_name]["vm_id"]
    return "unknown"


def _safe_resolve_path(root: Path, *parts: str) -> Path:
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts).resolve()
    candidate.relative_to(root_resolved)
    return candidate


# ================================================================
#  格式化函数（从 av_scan.py 迁移）
# ================================================================

def format_single_scan_result(scan_result: Dict, file_name: str, file_size: str) -> Dict:
    engines_result = []
    malicious_count = safe_count = unsupported_count = error_count = 0

    if "file_results" in scan_result and file_name in scan_result["file_results"]:
        file_result = scan_result["file_results"][file_name]
        for engine_name, detection in file_result["engines"].items():
            if detection == 1:
                status = "malicious"; malicious_count += 1
            elif detection == 0:
                status = "safe"; safe_count += 1
            else:
                status = "unsupported"; unsupported_count += 1

            label = file_result.get("labels", {}).get(engine_name, "")
            elapsed = 0
            for ed in scan_result.get("engine_details", []):
                if ed.get("engine") == engine_name:
                    elapsed = ed.get("elapsed_seconds", 0)
                    break
            engines_result.append({
                "name": engine_name, "status": status, "label": label,
                "vm": get_engine_vm(engine_name), "elapsed_seconds": round(elapsed, 3),
            })

    for ed in scan_result.get("engine_details", []):
        if not ed.get("success"):
            engines_result.append({
                "name": ed.get("engine"), "status": "error",
                "vm": ed.get("vm_id", "unknown"), "elapsed_seconds": 0,
                "error": ed.get("error", "未知错误"),
            })
            error_count += 1

    return {
        "file_name": file_name, "file_size": file_size,
        "scan_time": scan_result.get("scan_time", datetime.now().isoformat()),
        "elapsed_seconds": scan_result.get("elapsed_seconds", 0),
        "total_engines": scan_result.get("total_engines", 15),
        "malicious_count": malicious_count, "safe_count": safe_count,
        "unsupported_count": unsupported_count, "error_count": error_count,
        "engines": engines_result,
    }


def format_batch_scan_result(scan_result: Dict, file_name: str) -> Dict:
    engines = {}; malicious_count = 0
    if "file_results" in scan_result and file_name in scan_result["file_results"]:
        file_result = scan_result["file_results"][file_name]
        for engine_name, detection in file_result["engines"].items():
            label = file_result.get("labels", {}).get(engine_name, "")
            if detection == 1:
                engines[engine_name] = {"status": "malicious", "label": label}
                malicious_count += 1
            elif detection == 0:
                engines[engine_name] = {"status": "safe", "label": label}
            else:
                engines[engine_name] = {"status": "unsupported", "label": label}
    return {
        "file_name": file_name, "malicious_count": malicious_count,
        "safe_count": len(engines) - malicious_count, "engines": engines,
    }


# ================================================================
#  端点 1: 单文件同步扫描
# ================================================================

@router.post("/av_scan_single")
async def scan_single_file(file: UploadFile = File(...), engines: str = Form("")):
    if av_client is None:
        raise HTTPException(status_code=503, detail="AV服务未初始化")

    task_id = uuid.uuid4().hex[:8]
    join_queue(task_id)
    try:
        async with SCAN_LOCK:
            set_current({"type": "single_scan", "task_id": task_id, "file_name": file.filename})

            content = await file.read()
            file_size_str = f"{len(content) / (1024*1024):.2f} MB"
            logger.info(f"扫描: {file.filename} ({file_size_str})")
            selected_engines = [e.strip() for e in engines.split(",") if e.strip()] if engines else None

            scan_result = await asyncio.to_thread(
                av_client.scan_files,
                file_contents=[(file.filename, content)],
                engines=selected_engines,
            )
            formatted = format_single_scan_result(scan_result, file.filename, file_size_str)

            clear_current()
            return formatted
    except Exception as e:
        logger.error(f"扫描失败: {e}")
        raise HTTPException(status_code=500, detail=f"扫描失败: {e}")
    finally:
        leave_queue(task_id)


# ================================================================
#  端点 2: 单文件流式扫描 (SSE)
# ================================================================

@router.post("/av_scan_single_streaming")
async def scan_single_file_streaming(file: UploadFile = File(...), engines: str = Form("")):
    if av_client is None:
        raise HTTPException(status_code=503, detail="AV服务未初始化")

    raw_content = await file.read()
    file_size_str = f"{len(raw_content) / (1024*1024):.2f} MB"
    task_id = uuid.uuid4().hex[:8]
    selected_engines = [e.strip() for e in engines.split(",") if e.strip()] if engines else None
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
                yield f"data: {json.dumps({'type': 'queued', 'position': pos, 'message': f'排队中，前面还有 {pos-1} 个任务'})}\n\n"
                await asyncio.sleep(1)

            async with SCAN_LOCK:
                set_current({"type": "single_scan_streaming", "task_id": task_id, "file_name": file.filename})

                temp_dir = UPLOAD_DIR / "av_scan_temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                file_path = temp_dir / file.filename
                file_path.write_bytes(raw_content)

                yield f"data: {json.dumps({'type': 'file_info', 'file_name': file.filename, 'file_size': file_size_str})}\n\n"

                results = await asyncio.to_thread(
                    lambda: list(av_client.scan_single_file_streaming(
                        file_content=raw_content, file_name=file.filename,
                        engines=selected_engines,
                    ))
                )
                for r in results:
                    yield f"data: {json.dumps(r)}\n\n"

                if file_path.exists():
                    file_path.unlink()
                clear_current()
        except Exception as e:
            logger.error(f"流式扫描失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            leave_queue(task_id)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


# ================================================================
#  端点 3-7: 批量扫描
# ================================================================

@router.post("/av_batch_upload")
async def batch_upload_files(
    files: List[UploadFile] = File(...),
    engines: str = Form(""),
):
    task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    task_dir = _safe_resolve_path(BATCH_TASKS_ROOT, task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    for file in files:
        file_path = task_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        uploaded_files.append({
            "name": file.filename,
            "size": f"{len(content) / (1024*1024):.2f} MB",
            "path": str(file_path),
        })

    selected_engines = [e.strip() for e in engines.split(",") if e.strip()] if engines else AV_ENGINES

    batch_tasks[task_id] = {
        "status": "pending", "progress": 0.0, "total_files": len(files),
        "scanned_files": 0, "current_file": None, "start_time": datetime.now(),
        "files": uploaded_files, "results": [], "error": None,
        "selected_engines": selected_engines, "task_dir": str(task_dir),
    }

    logger.info(f"批量上传: task_id={task_id} files={len(files)} engines={len(selected_engines)}")
    return {
        "task_id": task_id, "total_files": len(files),
        "files": [{"name": f["name"], "size": f["size"]} for f in uploaded_files],
        "selected_engines": selected_engines,
    }


@router.post("/av_batch_scan_start")
async def start_batch_scan(request: dict, background_tasks: BackgroundTasks):
    if av_client is None:
        raise HTTPException(status_code=503, detail="AV服务未初始化")

    task_id = request.get("task_id")
    if not task_id or task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = batch_tasks[task_id]
    if task["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"任务状态不正确: {task['status']}")

    task["status"] = "queued" if SCAN_LOCK.locked() else "pending"
    join_queue(task_id)
    background_tasks.add_task(_execute_batch_scan, task_id)

    logger.info(f"批量扫描启动: task_id={task_id} status={task['status']}")
    return {"task_id": task_id, "status": task["status"],
            "message": "批量检测已启动" + ("（排队中）" if task["status"] == "queued" else "")}


async def _execute_batch_scan(task_id: str):
    try:
        task = batch_tasks[task_id]
        files = task["files"]
        selected_engines = task.get("selected_engines", AV_ENGINES)

        async with SCAN_LOCK:
            set_current({"type": "batch_scan", "task_id": task_id, "files": len(files), "engines": selected_engines})
            task["status"] = "running"
            task["start_time"] = datetime.now()

            async def scan_one(file_info, idx):
                try:
                    task["current_file"] = file_info["name"]
                    result = await asyncio.to_thread(
                        av_client.scan_single_file, file_info["path"],
                        selected_engines, len(selected_engines), 60,
                    )
                    formatted = format_batch_scan_result(result, file_info["name"])
                    task["scanned_files"] = idx + 1
                    task["progress"] = (idx + 1) / len(files) * 100
                    return formatted
                except Exception as e:
                    logger.error(f"扫描文件失败: {file_info['name']} {e}")
                    task["scanned_files"] = idx + 1
                    task["progress"] = (idx + 1) / len(files) * 100
                    return {"file_name": file_info["name"], "error": str(e), "engines": {}}

            max_concurrent = min(len(files), 5)
            results = []
            for batch_start in range(0, len(files), max_concurrent):
                batch_end = min(batch_start + max_concurrent, len(files))
                batch = [scan_one(files[i], i) for i in range(batch_start, batch_end)]
                results.extend(await asyncio.gather(*batch))

        task["results"] = results
        task["status"] = "completed"
        task["end_time"] = datetime.now()
        task["current_file"] = None
        logger.info(f"批量扫描完成: task_id={task_id}")

    except Exception as e:
        logger.error(f"批量扫描异常: task_id={task_id} {e}")
        task["status"] = "failed"
        task["error"] = str(e)
    finally:
        clear_current()
        leave_queue(task_id)


@router.get("/av_batch_scan_status/{task_id}")
async def get_batch_scan_status(task_id: str):
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = batch_tasks[task_id]
    elapsed = (datetime.now() - task["start_time"]).total_seconds()
    estimated_remaining = 0
    if task["scanned_files"] > 0 and task["scanned_files"] < task["total_files"]:
        avg = elapsed / task["scanned_files"]
        estimated_remaining = avg * (task["total_files"] - task["scanned_files"])
    return {
        "task_id": task_id, "status": task["status"],
        "progress": round(task["progress"], 2),
        "total_files": task["total_files"], "scanned_files": task["scanned_files"],
        "current_file": task.get("current_file"),
        "elapsed_seconds": round(elapsed, 2),
        "estimated_remaining": round(estimated_remaining, 2),
        "error": task.get("error"),
    }


@router.get("/av_batch_scan_result/{task_id}")
async def get_batch_scan_result(task_id: str):
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = batch_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务未完成: {task['status']}")
    return {
        "task_id": task_id, "status": task["status"],
        "scan_time": task["start_time"].isoformat(),
        "total_files": task["total_files"], "total_engines": 15,
        "results": task["results"],
    }


@router.get("/av_batch_scan_download/{task_id}")
async def download_batch_scan_report(task_id: str):
    if not BATCH_TASK_ID_RE.fullmatch(task_id or ""):
        raise HTTPException(status_code=400, detail="无效任务ID格式")
    if task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = batch_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务未完成: {task['status']}")

    task_dir = _safe_resolve_path(BATCH_TASKS_ROOT, Path(task.get("task_dir", "")).name)
    csv_path = task_dir / "report.csv"
    selected_engines = task.get("selected_engines", AV_ENGINES)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名"] + selected_engines)
        for result in task["results"]:
            row = [result["file_name"]]
            for engine in selected_engines:
                info = result["engines"].get(engine, {})
                status = info.get("status", "N/A") if isinstance(info, dict) else (info or "N/A")
                row.append({"malicious": "恶意", "safe": "安全", "unsupported": "不支持"}.get(status, status))
            writer.writerow(row)

    return FileResponse(str(csv_path), filename=f"av_scan_report_{task_id}.csv", media_type="text/csv")


# ================================================================
#  端点 8-10: 信息查询
# ================================================================

@router.get("/av_engines")
async def get_av_engines():
    if av_client is None:
        raise HTTPException(status_code=503, detail="AV服务未初始化")
    engines = av_client.get_available_engines()
    engine_list = [{"name": e, "vm": av_client.engine_to_vm.get(e, {}).get("vm_id", "unknown")} for e in engines]
    return {"total": len(engine_list), "engines": engine_list}


@router.get("/av_vm_status")
async def get_av_vm_status():
    if av_client is None:
        raise HTTPException(status_code=503, detail="AV服务未初始化")
    return av_client.get_vm_status()


@router.get("/av_scan_queue_status")
async def get_scan_queue_status():
    status = get_status()
    # 附加外部任务的元信息
    global _external_tasks
    if _external_tasks:
        status["external_tasks"] = list(_external_tasks.values())
    return status


@router.post("/av_scan_queue_register")
async def external_queue_register(request: dict):
    """外部服务（:5005 批量扫描）加入 :5006 内部排队，统一排队视图"""
    task_id = request.get("task_id", uuid.uuid4().hex[:8])
    task_type = request.get("type", "external")
    file_name = request.get("file_name", "unknown")
    file_count = request.get("file_count", 0)
    # 加入内部队列（所有扫描可见）
    pos = join_queue(task_id)
    # 同时存储元信息
    global _external_tasks
    _external_tasks[task_id] = {
        "type": task_type, "file_name": file_name, "file_count": file_count,
        "joined_at": datetime.now().isoformat(), "status": "queued", "position": pos,
    }
    return {"ok": True, "task_id": task_id, "position": pos}


@router.post("/av_scan_queue_unregister")
async def external_queue_unregister(request: dict):
    """外部服务（:5005 批量扫描）从队列移除"""
    task_id = request.get("task_id", "")
    leave_queue(task_id)
    global _external_tasks
    _external_tasks.pop(task_id, None)
    return {"ok": True}


@router.get("/av_scan_queue_position/{task_id}")
async def get_queue_position(task_id: str):
    """查询任务在 :5006 队列中的位置"""
    return {"position": get_position(task_id)}


# 存储外部注册任务的元信息
_external_tasks: Dict[str, Dict] = {}


# ================================================================
#  端点 11-12: 边界探测 (SSE)
# ================================================================

@router.post("/av_probe_start")
async def probe_start(file: UploadFile = File(...), engines: str = Form("")):
    """
    边界探测 — 二分查找定位检测边界，SSE 流式推送每步迭代。
    移植自 probe.py 的 SampleProber 状态机。
    """
    if av_client is None:
        raise HTTPException(status_code=503, detail="AV服务未初始化")

    raw_content = await file.read()
    selected_engines = [e.strip() for e in engines.split(",") if e.strip()] if engines else AV_ENGINES
    task_id = uuid.uuid4().hex[:12]
    join_queue(task_id)

    async def event_stream():
        try:
            # 排队等待
            while True:
                pos = get_position(task_id)
                if pos <= 0:
                    yield f"data: {json.dumps({'type': 'error', 'error': '任务已被取消'})}\n\n"
                    return
                if pos == 1:
                    break
                yield f"data: {json.dumps({'type': 'queued', 'position': pos, 'message': f'排队中，前面还有 {pos-1} 个任务'})}\n\n"
                await asyncio.sleep(1)

            async with SCAN_LOCK:
                set_current({"type": "probe", "task_id": task_id, "file_name": file.filename})

                yield f"data: {json.dumps({'type': 'start', 'probe_id': task_id, 'file_name': file.filename, 'file_size': len(raw_content), 'engines': selected_engines})}\n\n"

                all_results = []
                for engine in selected_engines:
                    yield f"data: {json.dumps({'type': 'engine_start', 'engine': engine})}\n\n"

                    try:
                        # 用 Queue 接收后台线程的实时进度事件
                        loop = asyncio.get_running_loop()
                        progress_q: asyncio.Queue = asyncio.Queue()

                        def on_progress(evt: dict):
                            loop.call_soon_threadsafe(progress_q.put_nowait, evt)

                        # 在后台线程中跑同步探测逻辑
                        async def _run_probe():
                            try:
                                return await asyncio.to_thread(
                                    _probe_engine_sync, raw_content, engine, task_id, on_progress,
                                )
                            finally:
                                loop.call_soon_threadsafe(progress_q.put_nowait, None)

                        probe_task = asyncio.create_task(_run_probe())

                        # 逐条转发进度事件，直到收到 None 哨兵
                        while True:
                            evt = await progress_q.get()
                            if evt is None:
                                break
                            yield f"data: {json.dumps(evt)}\n\n"

                        intervals, queries = await probe_task
                        all_results.append({
                            "engine": engine, "intervals": intervals,
                            "found_signatures": len(intervals),
                            "total_queries": queries,
                        })
                        yield f"data: {json.dumps({'type': 'engine_done', 'engine': engine, 'intervals': [[f'0x{a:x}', f'0x{b:x}'] for a,b in intervals], 'found_signatures': len(intervals), 'total_queries': queries})}\n\n"
                    except Exception as e:
                        logger.error(f"Probe {engine} error: {e}")
                        yield f"data: {json.dumps({'type': 'engine_error', 'engine': engine, 'error': str(e)})}\n\n"

                clear_current()
                yield f"data: {json.dumps({'type': 'complete', 'file_name': file.filename, 'file_size': len(raw_content), 'results': all_results})}\n\n"

        except Exception as e:
            logger.error(f"Probe 异常: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            leave_queue(task_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


MASK_BYTE = 0x90       # mask 字节
ZEROOUT_BYTE = 0x90    # bisecting 清零字节（与 mask 一致）
MAX_SIGNATURES = 20


def _mask_byte_for(original_byte: int) -> int:
    """若原字节已是 MASK_BYTE，则用 0x91 替代，确保 mask 后仍是非零。"""
    return 0x91 if original_byte == MASK_BYTE else MASK_BYTE


def _merge_intervals(intervals: list) -> list:
    """合并相邻/重叠区间。"""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for cur_start, cur_end in sorted_ivs[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))
    return merged


async def _probe_engine(data: bytes, engine: str, probe_id: str, on_progress=None) -> tuple:
    """对单个引擎执行二分边界探测（在后台线程中运行，避免阻塞事件循环）。"""
    return await asyncio.to_thread(_probe_engine_sync, data, engine, probe_id, on_progress)


def _probe_engine_sync(data: bytes, engine: str, probe_id: str, on_progress=None) -> tuple:
    """
    对单个引擎执行二分边界探测（同步版本，在线程中运行）。
    返回 (intervals: [(low, high), ...], total_queries: int)

    算法移植自 Boundary_Probing.py 的 SampleProcessor 状态机:
      initial → bisecting ⇄ masking_check → done

    关键点（与原版一致）：
    - 每次 bisecting 从原始数据出发，mask 已找到区间 + zero [mid, size)
    - mask 用 MASK_BYTE (0x90)，若原字节正好是 0x90 则用 0x91 替代
    - 只记录 1 字节边界点 (boundary-1, boundary)，最后合并
    """
    import random
    temp_dir = UPLOAD_DIR / "probe_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    size = len(data)
    intervals = []          # 已定位的边界点 [(start, end), ...]
    total_queries = 0
    found_signatures = 0

    # 二分搜索状态
    low = 0
    high = size
    mid = (low + high) // 2
    state = "initial"
    iteration = 0

    def scan_chunk(chunk: bytes) -> int:
        """传 bytes 直接扫描，返回 1=检出 0=未检出"""
        nonlocal total_queries
        total_queries += 1
        fname = f"{engine}_iter_{total_queries:04d}.bin"
        result = av_client.scan_single_file(
            file_content=chunk, file_name=fname,
            engines=[engine], timeout=60,
        )
        if "file_results" in result:
            if fname in result["file_results"]:
                engines_dict = result["file_results"][fname].get("engines", {})
                return 1 if engines_dict.get(engine, -1) == 1 else 0
        return 0

    def build_sample() -> bytes:
        """
        从原始数据出发，应用 mask 和 bisecting zero-out，返回待扫描的 bytes。
        与原始 SampleProcessor.prepare_workspace() 一致。
        """
        content = bytearray(data)

        # 1. mask 已找到的边界区间（用 0x90，碰巧为 0x90 则 0x91）
        for start, end in intervals:
            for i in range(start, end):
                content[i] = _mask_byte_for(content[i])

        # 2. bisecting 状态下清零后半段
        if state == "bisecting":
            for i in range(mid, len(content)):
                content[i] = ZEROOUT_BYTE

        return bytes(content)

    def _progress(extra=None):
        if on_progress:
            evt = {
                "type": "iteration", "engine": engine,
                "state": state, "iteration": iteration, "total_queries": total_queries,
                "found_signatures": found_signatures,
                "intervals_hex": [[f"0x{a:x}", f"0x{b:x}"] for a, b in intervals],
                "low": low, "high": high, "mid": mid, "detected": None,
            }
            if extra:
                evt.update(extra)
            on_progress(evt)

    # —— state machine ——
    while state not in ("done", "error"):
        iteration += 1
        prev_state = state

        if state == "initial":
            _progress({"detected": None})
            detected = scan_chunk(data)           # 扫描原始文件
            if detected:
                state = "bisecting"
                mid = (low + high) // 2
            else:
                _progress({"detected": False, "state": "clean"})
                state = "done"

        elif state == "bisecting":
            sample = build_sample()               # 原始 + mask + zero [mid, size)
            detected = scan_chunk(sample)
            _progress({"detected": bool(detected)})

            if detected:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2

            if low >= high - 1:                   # 二分收敛
                if found_signatures >= MAX_SIGNATURES:
                    state = "done"
                    break

                found_signatures += 1
                boundary = max(low, high)
                new_start = max(0, boundary - 1)
                new_end = boundary                # 1 字节边界点
                intervals = _merge_intervals(intervals + [(new_start, new_end)])

                state = "masking_check"

        elif state == "masking_check":
            sample = build_sample()               # 只 mask，不 zero
            detected = scan_chunk(sample)
            _progress({"detected": bool(detected)})

            if not detected:
                state = "done"                    # 全部特征已覆盖
            else:
                # 还有残留特征 → 重置二分，继续搜索
                low = 0
                high = size
                mid = (low + high) // 2
                iteration = 0
                state = "bisecting"

        if prev_state != state:
            _progress()

    return intervals, total_queries


@router.get("/av_probe_queue_status")
async def get_probe_queue_status():
    return get_status()


# ================================================================
#  健康检查
# ================================================================

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "av_detector",
        "vm_config": VM_CONFIG,
        "av_client_ready": av_client is not None,
    }


# ================================================================
#  启动
# ================================================================

@app.on_event("startup")
async def startup():
    global av_client
    logger.info(f"AV Detector Service starting ...")
    logger.info(f"VM config: {VM_CONFIG}")

    try:
        av_client = AVDistributedClient(config_path=VM_CONFIG)
        engines = av_client.get_available_engines()
        logger.info(f"AV client OK, {len(engines)} engines loaded")

        # 快速健康检查
        vm_status = av_client.get_vm_status()
        online = sum(1 for v in vm_status.values() if v)
        logger.info(f"VM status: {online}/{len(vm_status)} online")
        for vm_key, ok in vm_status.items():
            logger.info(f"  {'🟢' if ok else '🔴'} {vm_key}")

    except Exception as e:
        logger.error(f"AV client init failed: {e}")
        logger.warning("Service started but AV scanning will not work until config is fixed")


app.include_router(router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
