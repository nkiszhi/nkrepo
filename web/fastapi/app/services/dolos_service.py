"""
Dolos代码同源性检测服务层 - Docker版本
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from dataclasses import dataclass
import uuid
import logging
import httpx
import os
import re
import ipaddress
import socket
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

ALLOWED_DOLOS_URL_SCHEMES = {"https"}
BLOCKED_DOLOS_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
}


@dataclass(frozen=True)
class SafeDownloadURL:
    """URL value that has passed SSRF validation."""

    url: str
    filename: str


def _is_public_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


def _validate_public_host(hostname: str) -> str:
    if not hostname:
        raise ValueError("URL hostname is required")

    normalized = hostname.rstrip(".").lower()
    if normalized in BLOCKED_DOLOS_HOSTNAMES or normalized.endswith(".localhost"):
        raise ValueError("Internal hostnames are not allowed")

    try:
        ip_obj = ipaddress.ip_address(normalized)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        if not ip_obj.is_global:
            raise ValueError("Only public IP addresses are allowed")
        return normalized

    try:
        address_infos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve hostname: {hostname}") from exc

    resolved_ips = {info[4][0] for info in address_infos}
    if not resolved_ips:
        raise ValueError(f"Unable to resolve hostname: {hostname}")
    if any(not _is_public_ip(ip) for ip in resolved_ips):
        raise ValueError("Host resolves to a non-public address")

    return normalized


def _allowed_dolos_base_url(hostname: str) -> str:
    if hostname == "raw.githubusercontent.com":
        return "https://raw.githubusercontent.com"
    if hostname == "gist.githubusercontent.com":
        return "https://gist.githubusercontent.com"
    if hostname == "github.com":
        return "https://github.com"
    if hostname == "gitlab.com":
        return "https://gitlab.com"
    if hostname == "bitbucket.org":
        return "https://bitbucket.org"
    if hostname == "gitee.com":
        return "https://gitee.com"
    raise ValueError(f"URL host is not allowed: {hostname}")


def _validate_dolos_path(path: str) -> str:
    normalized_path = path or "/"
    decoded_path = unquote(normalized_path)

    if not normalized_path.startswith("/"):
        raise ValueError("URL path must be absolute")
    if "\\" in decoded_path or "\x00" in decoded_path:
        raise ValueError("URL path contains invalid characters")
    if "/../" in decoded_path or decoded_path.startswith("/../") or decoded_path.endswith("/.."):
        raise ValueError("URL path traversal is not allowed")
    if not re.fullmatch(r"/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*", normalized_path):
        raise ValueError("URL path contains invalid characters")

    return normalized_path


def _validate_dolos_query(query: str) -> str:
    if len(query) > 2048:
        raise ValueError("URL query is too long")
    if query and not re.fullmatch(r"[A-Za-z0-9._~!$&'()*+,;=:@%/?-]*", query):
        raise ValueError("URL query contains invalid characters")
    return query


def _validate_dolos_download_url(url: str) -> SafeDownloadURL:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL")

    scheme = parsed.scheme.lower()
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are not allowed")

    if scheme not in ALLOWED_DOLOS_URL_SCHEMES:
        raise ValueError("Only HTTPS URLs are allowed")

    hostname = _validate_public_host(parsed.hostname)
    base_url = _allowed_dolos_base_url(hostname)

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL port") from exc

    if port not in (None, 443):
        raise ValueError("Only the default HTTPS port is allowed")

    safe_path = _validate_dolos_path(parsed.path)
    safe_query = _validate_dolos_query(parsed.query)
    safe_path_and_query = safe_path
    if safe_query:
        safe_path_and_query = f"{safe_path}?{safe_query}"

    normalized_url = f"{base_url}{safe_path_and_query}"

    filename = Path(unquote(parsed.path)).name
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    if not filename or filename in {".", ".."}:
        filename = f"file_{uuid.uuid4().hex}"

    return SafeDownloadURL(url=normalized_url, filename=filename)


class DolosAnalyzer:
    """Dolos分析器 - Docker版本"""
    
    def __init__(self):
        self.results_dir = Path("dolos_results")
        self.results_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("uploads/dolos_temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def analyze(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        执行Dolos分析
        
        Args:
            file_paths: 文件路径列表
        
        Returns:
            分析结果字典
        """
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # 输出文件路径
        output_file = self.results_dir / f"{analysis_id}.json"
        
        # 获取文件所在目录
        file_dir = Path(file_paths[0]).parent.absolute()
        
        # 构建Docker命令
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{file_dir}:/data",
            "ghcr.io/dodona-edu/dolos:latest",
            "run",
        ]
        
        # 添加文件路径（相对于挂载点）
        for file_path in file_paths:
            file_name = Path(file_path).name
            cmd.append(f"/data/{file_name}")
        
        logger.info(f"执行Dolos分析: {' '.join(cmd)}")
        
        # 异步执行命令
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "未知错误"
            logger.error(f"Dolos分析失败: {error_msg}")
            raise Exception(f"Dolos分析失败: {error_msg}")
        
        # 从stdout解析结果
        try:
            output = stdout.decode()
            result = self._parse_dolos_output(output, file_paths)
        except Exception as e:
            logger.error(f"解析Dolos结果失败: {str(e)}")
            raise Exception(f"解析分析结果失败: {str(e)}")
        
        # 添加元数据
        result['analysis_id'] = analysis_id
        result['timestamp'] = timestamp
        result['files'] = [Path(p).name for p in file_paths]
        
        # 保存完整结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dolos分析完成: {analysis_id}")
        return result
    
    def _parse_dolos_output(self, output: str, file_paths: List[str]) -> Dict[str, Any]:
        """
        解析Dolos终端输出
        
        Args:
            output: 终端输出字符串
            file_paths: 文件路径列表
        
        Returns:
            标准化的结果字典
        """
        pairs = []
        
        # 解析相似度分数
        similarity_match = re.search(r'Similarity score:\s+(\d+)', output)
        similarity = int(similarity_match.group(1)) / 100 if similarity_match else 0
        
        # 为每对文件创建结果
        for i in range(len(file_paths)):
            for j in range(i + 1, len(file_paths)):
                pairs.append({
                    'leftFile': file_paths[i],
                    'rightFile': file_paths[j],
                    'similarity': similarity,
                    'overlap': 0
                })
        
        return {'pairs': pairs}
    
    async def get_history(
        self,
        skip: int = 0,
        limit: int = 10,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取分析历史
        
        Args:
            skip: 跳过记录数
            limit: 返回记录数
            search: 搜索关键词
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            历史记录字典
        """
        results = []
        all_files = sorted(self.results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        for file in all_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 搜索过滤
                if search:
                    search_lower = search.lower()
                    if (search_lower not in data.get('analysis_id', '').lower() and
                        not any(search_lower in str(f).lower() for f in data.get('files', []))):
                        continue
                
                # 日期过滤
                timestamp = data.get('timestamp', '')
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date + ' 23:59:59':
                    continue
                
                results.append({
                    'analysis_id': data.get('analysis_id'),
                    'timestamp': data.get('timestamp'),
                    'files': data.get('files', []),
                    'file_count': len(data.get('files', [])),
                    'pair_count': len(data.get('pairs', []))
                })
            except Exception as e:
                logger.error(f"读取历史记录失败 {file}: {str(e)}")
                continue
        
        total = len(results)
        items = results[skip:skip+limit]
        
        return {
            'items': items,
            'total': total,
            'skip': skip,
            'limit': limit
        }
    
    async def get_result(self, analysis_id: str) -> Optional[Dict]:
        """
        获取特定分析结果
        
        Args:
            analysis_id: 分析ID
        
        Returns:
            分析结果字典
        """
        result_file = self.results_dir / f"{analysis_id}.json"
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取分析结果失败: {str(e)}")
            return None
    
    async def delete_result(self, analysis_id: str) -> bool:
        """
        删除分析结果
        
        Args:
            analysis_id: 分析ID
        
        Returns:
            是否删除成功
        """
        result_file = self.results_dir / f"{analysis_id}.json"
        if not result_file.exists():
            return False
        
        try:
            os.remove(result_file)
            logger.info(f"删除分析结果: {analysis_id}")
            return True
        except Exception as e:
            logger.error(f"删除分析结果失败: {str(e)}")
            return False
    
    async def analyze_from_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        从URL分析代码文件
        
        Args:
            urls: 文件URL列表
        
        Returns:
            分析结果
        """
        # Validate all URLs before making any outbound request.
        validated_urls: List[SafeDownloadURL] = []
        for url in urls:
            try:
                safe_url = _validate_dolos_download_url(url)
            except ValueError:
                logger.error(f"URL验证失败: {url}")
                raise ValueError(f"无效或不安全的URL: {url}")
            validated_urls.append(safe_url)
        
        # 下载文件
        file_paths = []
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            for index, safe_url in enumerate(validated_urls):
                try:
                    response = await client.get(
                        safe_url.url,
                        follow_redirects=False,
                    )
                    if response.is_redirect:
                        raise ValueError("URL重定向被禁止")
                    response.raise_for_status()
                    
                    # 保存文件
                    filename = f"{index}_{safe_url.filename}"
                    file_path = self.temp_dir / filename
                    
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    file_paths.append(str(file_path))
                except Exception as e:
                    logger.error(f"下载文件失败 {safe_url.url}: {str(e)}")
                    raise Exception(f"下载文件失败: {safe_url.url}")
        
        # 执行分析
        result = await self.analyze(file_paths)
        
        # 清理临时文件
        for path in file_paths:
            try:
                os.remove(path)
            except Exception:
                pass
        
        return result

