"""
Web 后端 → 独立检测服务 (:5006) HTTP 客户端
替换原来直接 import AVDistributedClient 的方式，所有扫描请求转发到独立服务。
"""

import json
import os
import requests
from typing import Dict, List, Generator, Optional

# 独立检测服务地址（部署时按实际 IP 修改）
AV_SERVICE_URL = os.environ.get("AV_SERVICE_URL", "http://127.0.0.1:5006")

# 如果独立服务部署在中关村主机上，改成对应 IP：
# AV_SERVICE_URL = "http://192.168.8.202:5006"


class AVServiceClient:
    """HTTP 客户端，封装对独立检测服务 :5006 的调用"""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or AV_SERVICE_URL).rstrip("/")

    # ---- 同步扫描 ----

    def scan_single(self, file_path: str) -> Dict:
        """单文件同步扫描 → JSON 结果"""
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/api/av_scan_single",
                files={"file": f},
                timeout=600,
            )
        resp.raise_for_status()
        return resp.json()

    # ---- 流式扫描 (SSE) ----

    def scan_single_streaming(self, file_path: str) -> Generator[Dict, None, None]:
        """单文件流式扫描 → SSE 事件流生成器"""
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{self.base_url}/api/av_scan_single_streaming",
                files={"file": f},
                stream=True,
                timeout=600,
            )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                data_str = line[6:].decode("utf-8", errors="replace")
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    continue

    # ---- 批量上传 ----

    def batch_upload(self, file_paths: List[str], engines: List[str] = None) -> Dict:
        """批量上传文件 → 返回 task_id"""
        files = []
        opened = []
        try:
            for path in file_paths:
                f = open(path, "rb")
                opened.append(f)
                files.append(("files", (f.name if hasattr(f, "name") else path, f, "application/octet-stream")))
            data = {}
            if engines:
                data["engines"] = ",".join(engines)
            resp = requests.post(
                f"{self.base_url}/api/av_batch_upload",
                files=files,
                data=data,
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()
        finally:
            for f in opened:
                f.close()

    # ---- 批量扫描控制 ----

    def batch_scan_start(self, task_id: str) -> Dict:
        resp = requests.post(
            f"{self.base_url}/api/av_batch_scan_start",
            json={"task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def batch_scan_status(self, task_id: str) -> Dict:
        resp = requests.get(
            f"{self.base_url}/api/av_batch_scan_status/{task_id}",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def batch_scan_result(self, task_id: str) -> Dict:
        resp = requests.get(
            f"{self.base_url}/api/av_batch_scan_result/{task_id}",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def batch_scan_download(self, task_id: str) -> bytes:
        resp = requests.get(
            f"{self.base_url}/api/av_batch_scan_download/{task_id}",
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content

    # ---- 信息查询 ----

    def get_engines(self) -> Dict:
        resp = requests.get(f"{self.base_url}/api/av_engines", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_vm_status(self) -> Dict:
        resp = requests.get(f"{self.base_url}/api/av_vm_status", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_queue_status(self) -> Dict:
        resp = requests.get(f"{self.base_url}/api/av_scan_queue_status", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ---- 边界探测 (SSE) ----

    def probe_start(self, file_path: str, engines: List[str] = None) -> Generator[Dict, None, None]:
        """边界探测 → SSE 事件流生成器"""
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {}
            if engines:
                data["engines"] = ",".join(engines)
            resp = requests.post(
                f"{self.base_url}/api/av_probe_start",
                files=files,
                data=data,
                stream=True,
                timeout=3600,
            )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                data_str = line[6:].decode("utf-8", errors="replace")
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    continue

    def probe_queue_status(self) -> Dict:
        resp = requests.get(f"{self.base_url}/api/av_probe_queue_status", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ---- 健康检查 ----

    def health(self) -> Dict:
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()


# 模块级单例
av_service_client = AVServiceClient()
