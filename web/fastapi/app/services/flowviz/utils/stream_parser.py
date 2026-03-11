#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
流式JSON解析器 - 实时解析AI返回的JSON片段
"""
import json
import re
import logging
from typing import Dict, List, Optional, Generator

logger = logging.getLogger(__name__)

class StreamingJSONParser:
    """流式JSON解析器，实时提取节点和边"""
    
    def __init__(self):
        self.buffer = ""
        self.in_object = False
        self.brace_depth = 0
        self.current_object = ""
        self.expecting_nodes = False
        self.expecting_edges = False
        
    def feed(self, chunk: str) -> Generator[Dict, None, None]:
        """喂入文本片段，产生解析结果"""
        self.buffer += chunk
        
        # 尝试提取JSON对象
        objects = self.extract_json_objects()
        
        for obj_str in objects:
            try:
                data = json.loads(obj_str)
                
                # 提取节点
                if "nodes" in data and isinstance(data["nodes"], list):
                    for node in data["nodes"]:
                        if isinstance(node, dict):
                            yield {"type": "node", "node": node}
                
                # 提取边
                if "edges" in data and isinstance(data["edges"], list):
                    for edge in data["edges"]:
                        if isinstance(edge, dict):
                            yield {"type": "edge", "edge": edge}
                            
                # 如果是完整的响应
                if "nodes" in data or "edges" in data:
                    yield {"type": "partial_result", "data": data}
                    
            except json.JSONDecodeError as e:
                # 尝试提取不完整的节点/边
                extracted = self.extract_partial_objects(obj_str)
                for item in extracted:
                    yield item
    
    def extract_json_objects(self) -> List[str]:
        """从缓冲区提取可能的JSON对象"""
        objects = []
        
        # 查找{和}的匹配
        start = self.buffer.find('{')
        while start != -1:
            brace_count = 0
            in_string = False
            escape = False
            
            for i in range(start, len(self.buffer)):
                char = self.buffer[i]
                
                if escape:
                    escape = False
                    continue
                    
                if char == '\\':
                    escape = True
                    continue
                    
                if char == '"' and not escape:
                    in_string = not in_string
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        
                        if brace_count == 0:
                            # 找到一个完整的对象
                            obj_str = self.buffer[start:i+1]
                            try:
                                # 验证是有效的JSON
                                json.loads(obj_str)
                                objects.append(obj_str)
                                # 移除已处理的部分
                                self.buffer = self.buffer[i+1:]
                                start = self.buffer.find('{')
                                break
                            except:
                                # 不是有效JSON，继续查找
                                pass
                
                if i == len(self.buffer) - 1:
                    # 到缓冲区末尾也没找到完整对象
                    return objects
                    
            if start == -1:
                break
                
        return objects
    
    def extract_partial_objects(self, text: str) -> List[Dict]:
        """提取不完整的JSON对象中的节点和边"""
        results = []
        
        # 尝试提取节点模式
        node_pattern = r'\{\s*"id"\s*:\s*"([^"]+)"[^}]*"type"\s*:\s*"([^"]+)"[^}]*\}(?=\s*,|\s*\])'
        node_matches = re.findall(node_pattern, text, re.DOTALL)
        
        for node_match in node_matches:
            node_id, node_type = node_match
            # 尝试构建节点对象
            node_start = text.find(f'"id": "{node_id}"')
            if node_start != -1:
                # 查找节点结束
                node_end = text.find('}', node_start)
                if node_end != -1:
                    node_text = text[node_start-1:node_end+1]
                    try:
                        node = json.loads(node_text)
                        results.append({"type": "node", "node": node})
                    except:
                        pass
        
        # 尝试提取边模式
        edge_pattern = r'\{\s*"id"\s*:\s*"([^"]+)"[^}]*"source"\s*:\s*"([^"]+)"[^}]*"target"\s*:\s*"([^"]+)"[^}]*\}'
        edge_matches = re.findall(edge_pattern, text, re.DOTALL)
        
        for edge_match in edge_matches:
            edge_id, source, target = edge_match
            # 尝试构建边对象
            edge_start = text.find(f'"id": "{edge_id}"')
            if edge_start != -1:
                edge_end = text.find('}', edge_start)
                if edge_end != -1:
                    edge_text = text[edge_start-1:edge_end+1]
                    try:
                        edge = json.loads(edge_text)
                        results.append({"type": "edge", "edge": edge})
                    except:
                        pass
        
        return results
    
    def reset(self):
        """重置解析器状态"""
        self.buffer = ""
        self.in_object = False
        self.brace_depth = 0
        self.current_object = ""