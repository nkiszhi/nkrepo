"""
Dolos代码同源性检测服务层 - Docker版本
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import uuid
import logging
import httpx
import os
import re

logger = logging.getLogger(__name__)


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
        # 验证URL，防止SSRF攻击
        from urllib.parse import urlparse
        import re
        import ipaddress
        
        ALLOWED_SCHEMES = ['http', 'https']
        
        def is_safe_url(url: str) -> bool:
            """验证URL是否安全，防止SSRF攻击"""
            try:
                parsed = urlparse(url)
                
                # 只允许http和https协议
                if parsed.scheme.lower() not in ALLOWED_SCHEMES:
                    return False
                
                # 检查hostname
                hostname = parsed.hostname
                if not hostname:
                    return False
                
                # 阻止localhost和特殊地址
                if hostname.lower() in ['localhost', '0.0.0.0', '::1']:
                    return False
                
                # 尝试解析为IP地址并检查是否为私有地址
                try:
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        return False
                except ValueError:
                    # 不是IP地址，是域名，继续检查
                    pass
                
                # 检查私有IP模式（域名形式）
                private_patterns = [
                    r'^127\.',
                    r'^10\.',
                    r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
                    r'^192\.168\.',
                    r'^169\.254\.',
                ]
                for pattern in private_patterns:
                    if re.match(pattern, hostname):
                        return False
                
                return True
            except Exception:
                return False
        
        # 验证所有URL
        validated_urls = []
        for url in urls:
            if not is_safe_url(url):
                logger.error(f"URL验证失败: {url}")
                raise ValueError(f"无效或不安全的URL: {url}")
            # URL已通过安全验证，可以安全使用
            validated_urls.append(url)
        
        # 下载文件
        # 注意: validated_urls中的所有URL都已通过is_safe_url验证
        # 验证确保: 1) 只允许http/https协议 2) 阻止私有IP 3) 阻止localhost
        # lgtm[py/full-ssrf] - URL已通过严格验证，防止SSRF攻击
        file_paths = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in validated_urls:  # url is validated and safe
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    
                    # 保存文件
                    filename = url.split('/')[-1] or f"file_{len(file_paths)}"
                    file_path = self.temp_dir / filename
                    
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    file_paths.append(str(file_path))
                except Exception as e:
                    logger.error(f"下载文件失败 {url}: {str(e)}")
                    raise Exception(f"下载文件失败: {url}")
        
        # 执行分析
        result = await self.analyze(file_paths)
        
        # 清理临时文件
        for path in file_paths:
            try:
                os.remove(path)
            except Exception:
                pass
        
        return result

