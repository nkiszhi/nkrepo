"""
边界探测 API — 使用二分查找定位杀毒引擎的检测边界

算法移植自 Boundary_Probing.py 的 SampleProcessor 状态机：
  initial → bisecting ⇄ masking_check → done

每个引擎独立运行：上传样本 → 二分法清零字节 → 调用 VM 扫描 → 收敛到字节级边界
通过 SSE (Server-Sent Events) 流式返回每步进度。

任务排队：同一时间只允许一个探测任务运行，后续请求自动排队等待。
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from fastapi.responses import StreamingResponse
from app.api.auth import get_current_user
from app.core import settings
from pathlib import Path
import os
import logging
import json
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

# 新版 probe 通过 :5006 独立检测服务代理执行，不在 Web 启动时初始化本地 AV 客户端。
av_client = None

# ========== 常量 ==========
ZEROOUT_BYTE = 0x90      # 清零字节
MASK_BYTE = 0x90         # mask 字节
MASK_ALT = 0x91          # mask 替代字节（当原字节正好是 0x90 时）
MAX_SIGNATURES = 20      # 单个文件最多找 20 个特征区间
AV_ENGINES = [
    "Avira", "McAfee", "WindowsDefender", "IkarusT3", "Emsisoft",
    "FProtect", "Vba32", "ClamAV", "Kaspersky", "ESET",
    "DrWeb", "Avast", "AVG", "AdAware", "FSecure",
]

# ========== 任务排队（共享模块，与 av_scan.py 共用同一把锁）==========
from app.services.av_detection.scan_queue import (
    SCAN_LOCK, join_queue, leave_queue, get_position, set_current, clear_current, get_status
)

# ========== SSE 辅助函数 ==========


def _sse(data: dict) -> str:
    """将 dict 编码为 SSE 格式的字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_error(engine: str, message: str) -> str:
    return _sse({"type": "engine_error", "engine": engine, "error": message})


# ========== 边界探测核心类 ==========


class SampleProber:
    """单个文件 + 单个引擎的边界探测器。

    状态机:
        initial → bisecting → masking_check → done
    """

    def __init__(self, original_data: bytes, engine: str, temp_dir: Path):
        self.original = bytearray(original_data)
        self.size = len(original_data)
        self.engine = engine
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.low = 0
        self.high = self.size
        self.mid = (self.low + self.high) // 2
        self.state = "initial"
        self.iteration = 0
        self.intervals: list = []       # [(start, end), ...] 已定位的边界区间
        self.found_signatures = 0
        self.total_queries = 0
        self._last_error: str = ""

    # ---------- 文件准备 ----------

    def _apply_mask(self, data: bytearray, start: int, end: int):
        """对 [start, end) 区间做 mask 处理（避免和清零区混淆）。"""
        for i in range(start, end):
            data[i] = MASK_ALT if data[i] == MASK_BYTE else MASK_BYTE

    def write_sample(self) -> Path:
        """按当前状态生成样本文件：清零 [mid, size) + mask 已找到区间，返回文件路径。"""
        data = bytearray(self.original)

        # 1. mask 已找到的边界区间
        for start, end in self.intervals:
            self._apply_mask(data, start, end)

        # 2. bisecting 状态下清零后半段
        if self.state == "bisecting":
            for i in range(self.mid, len(data)):
                data[i] = ZEROOUT_BYTE

        file_path = self.temp_dir / f"{self.engine}_iter_{self.iteration:04d}.bin"
        file_path.write_bytes(bytes(data))
        return file_path

    # ---------- 状态机 ----------

    def update(self, detected: bool) -> dict:
        """根据检测结果推进状态机，返回当前状态快照。"""
        self.iteration += 1
        self.total_queries += 1

        prev_state = self.state

        if self.state == "initial":
            self._handle_initial(detected)
        elif self.state == "bisecting":
            self._handle_bisecting(detected)
        elif self.state == "masking_check":
            self._handle_masking_check(detected)

        if prev_state != self.state:
            logger.info(
                f"[Probe] {self.engine}: {prev_state} → {self.state}  "
                f"iter={self.iteration}  low=0x{self.low:X}  high=0x{self.high:X}"
            )

        return self.status()

    def _handle_initial(self, detected: bool):
        if detected:
            self.state = "bisecting"
            self.mid = (self.low + self.high) // 2
        else:
            self.state = "done"

    def _handle_bisecting(self, detected: bool):
        if detected:
            self.high = self.mid
        else:
            self.low = self.mid

        self.mid = (self.low + self.high) // 2

        if self.low >= self.high - 1:
            if self.found_signatures >= MAX_SIGNATURES:
                self.state = "done"
                return
            self.found_signatures += 1
            boundary = max(self.low, self.high)
            self._record_boundary(boundary)
            self.state = "masking_check"

    def _handle_masking_check(self, detected: bool):
        if detected:
            self._reset_search()
            self.state = "bisecting"
        else:
            self.state = "done"

    def _record_boundary(self, boundary: int):
        """记录一个边界点，与已有区间合并。"""
        new_start = max(0, boundary - 1)
        new_end = boundary
        merged = self.intervals + [(new_start, new_end)]
        merged.sort(key=lambda x: x[0])
        result = [merged[0]]
        for cur_s, cur_e in merged[1:]:
            prev_s, prev_e = result[-1]
            if cur_s <= prev_e + 1:
                result[-1] = (prev_s, max(prev_e, cur_e))
            else:
                result.append((cur_s, cur_e))
        self.intervals = result

    def _reset_search(self):
        self.low = 0
        self.high = self.size
        self.mid = (self.low + self.high) // 2
        self.iteration = 0

    # ---------- 状态查询 ----------

    def status(self) -> dict:
        return {
            "engine": self.engine,
            "state": self.state,
            "iteration": self.iteration,
            "low": self.low,
            "high": self.high,
            "mid": self.mid,
            "size": self.size,
            "intervals": self.intervals,
            "intervals_hex": [(hex(s), hex(e)) for s, e in self.intervals],
            "found_signatures": self.found_signatures,
            "total_queries": self.total_queries,
            "error": self._last_error,
        }

    @property
    def is_done(self) -> bool:
        return self.state in ("done", "error")


# ========== 内部：运行一次完整探测 ==========


async def _run_probe_engines(raw_data: bytes, selected: List[str],
                              probe_dir: Path) -> List[dict]:
    """对原始数据依次运行选中的引擎，返回每个引擎的最终结果。"""
    all_results = []

    for engine in selected:
        prober = SampleProber(raw_data, engine, probe_dir)
        try:
            while not prober.is_done:
                sample_path = prober.write_sample()
                try:
                    result = await asyncio.to_thread(
                        av_client.scan_single_file,
                        str(sample_path),
                        [engine],
                        1,
                        30,
                    )
                    detected = _extract_detection(result, engine)
                except Exception as scan_err:
                    logger.error(f"[Probe] {engine} 扫描异常: {scan_err}")
                    prober._last_error = str(scan_err)
                    detected = False

                status = prober.update(detected)

                # # 推送事件通过 SSE 迭代器 —— 这里不直接 yield，
                # 而是返回状态让外层 SSE generator 处理
                # (由 event_stream 里的代码负责 yield)

                if sample_path.exists():
                    sample_path.unlink()

            all_results.append(prober.status())

        except Exception as engine_err:
            logger.error(f"[Probe] {engine} 探测异常: {engine_err}")
            all_results.append({
                "engine": engine,
                "state": "error",
                "error": str(engine_err),
                "found_signatures": 0,
                "total_queries": prober.total_queries,
                "intervals": [],
                "intervals_hex": [],
            })

    return all_results


# ========== API 端点 ==========


AV_PROBE_SERVICE_URL = getattr(settings, "AV_SERVICE_URL", os.environ.get("AV_SERVICE_URL", "http://127.0.0.1:5006"))


@router.get("/av_probe_queue_status")
async def get_probe_queue_status(current_user: dict = Depends(get_current_user)):
    """查询当前扫描队列状态（从独立服务查询）"""
    try:
        import requests
        def _call():
            resp = requests.get(f"{AV_PROBE_SERVICE_URL}/api/av_probe_queue_status", timeout=10)
            resp.raise_for_status()
            return resp.json()
        return await asyncio.to_thread(_call)
    except requests.RequestException:
        return get_status()


@router.post("/av_probe_start")
async def start_probe(
    file: UploadFile = File(...),
    engines: str = Form(""),
    current_user: dict = Depends(get_current_user),
):
    """启动边界探测任务（代理 SSE 流从独立检测服务 :5006）"""
    import requests

    raw_data = await file.read()

    async def event_stream():
        import threading
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()

        def _stream():
            try:
                _data = {}
                if engines:
                    _data["engines"] = engines
                resp = requests.post(
                    f"{AV_PROBE_SERVICE_URL}/api/av_probe_start",
                    files={"file": (file.filename, raw_data, file.content_type or "application/octet-stream")},
                    data=_data,
                    stream=True,
                    timeout=3600,
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
                    q.put_nowait, _sse({"type": "error", "error": str(e)})
                )
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=_stream, daemon=True).start()

        while True:
            chunk = await q.get()
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


def _extract_detection(result: dict, engine: str) -> bool:
    """从 scan_single_file 的返回结果中提取某引擎的检测结果（True=恶意）。"""
    if not result or "file_results" not in result:
        return False

    for file_name, file_result in result["file_results"].items():
        detection = file_result.get("engines", {}).get(engine, -1)
        return detection == 1

    return False
