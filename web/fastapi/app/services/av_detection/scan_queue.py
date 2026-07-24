"""
AV 扫描任务排队模块

所有 VM 扫描操作（probe / single / batch / streaming）共享同一个互斥锁，
确保同一时间只有一个扫描任务在执行，避免 VM 过载。
"""

import asyncio
from typing import List, Dict
from datetime import datetime

# ====== 共享锁 — 所有扫描端点的唯一互斥锁 ======
SCAN_LOCK = asyncio.Lock()

# ====== 排队队列 ======
_queue: List[str] = []

# ====== 当前任务信息 ======
_current: Dict = {}


def join_queue(task_id: str) -> int:
    """加入排队队列，返回当前队列长度（含自己）。"""
    _queue.append(task_id)
    return len(_queue)


def leave_queue(task_id: str):
    """从队列中移除。"""
    if task_id in _queue:
        _queue.remove(task_id)


def get_position(task_id: str) -> int:
    """获取排队位置（1 = 轮到你了），不在队列中返回 -1。"""
    try:
        return _queue.index(task_id) + 1
    except ValueError:
        return -1


def set_current(info: Dict):
    """设置当前正在执行的任务信息。"""
    global _current
    _current = info
    _current["started_at"] = datetime.now().isoformat()


def clear_current():
    """清除当前任务信息。"""
    global _current
    _current = {}


def get_status() -> Dict:
    """获取当前队列状态。"""
    return {
        "running": SCAN_LOCK.locked(),
        "queue_length": max(0, len(_queue) - (1 if SCAN_LOCK.locked() else 0)),
        "current": _current if SCAN_LOCK.locked() else None,
        "all_queued": list(_queue),
    }
