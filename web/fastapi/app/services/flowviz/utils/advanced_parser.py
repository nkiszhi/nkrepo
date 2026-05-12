#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
Advanced Streaming Parser - 简化版，更灵活的验证
"""
import json
import re
import logging
from typing import Dict, List, Any, Optional
import traceback
import os

logger = logging.getLogger(__name__)

# 调试模式开关 - 生产环境应设置为False
DEBUG_MODE = os.environ.get('FLOWVIZ_DEBUG', 'false').lower() == 'true'

class AdvancedFlowParser:
    """Advanced Streaming Parser - 简化版，更灵活的验证"""
    
    def __init__(self, request_id):
        self.request_id = request_id
    
    def parse_technical_data(self, raw_text: str) -> Dict[str, Any]:
        """Parse AI response - 灵活的JSON解析"""
        logger.info(f"🔍 [{self.request_id}] 解析AI响应，长度: {len(raw_text)} 字符")
        
        if not raw_text or len(raw_text.strip()) < 100:
            logger.error(f"❌ [{self.request_id}] 响应文本太短")
            return {'error': 'AI响应太短，无法解析'}
        
        # Save debug file
        self.save_debug_file(raw_text)
        
        # 先清理文本中的控制字符
        cleaned_text = self.clean_control_characters(raw_text)
        
        # 尝试直接JSON解析
        result = self.parse_json_flexible(cleaned_text)
        if result:
            logger.info(f"✅ [{self.request_id}] JSON解析成功: {len(result.get('nodes', []))} 个节点, {len(result.get('edges', []))} 条边")
            return result
        
        # 尝试从代码块提取JSON
        result = self.extract_json_from_anywhere(cleaned_text)
        if result:
            logger.info(f"✅ [{self.request_id}] 从代码块提取JSON成功: {len(result.get('nodes', []))} 个节点, {len(result.get('edges', []))} 条边")
            return result
        
        # 如果所有解析都失败，返回错误
        error_preview = cleaned_text[:500] if len(cleaned_text) > 500 else cleaned_text
        logger.error(f"❌ [{self.request_id}] 无法将AI响应解析为JSON")
        
        return {
            'error': 'AI响应不是有效的JSON格式。请确保AI返回正确的JSON格式。',
            'raw_preview': cleaned_text[:1000] if len(cleaned_text) > 1000 else cleaned_text,
            'raw_length': len(cleaned_text)
        }
    
    def clean_control_characters(self, text: str) -> str:
        """清理控制字符"""
        if not text:
            return text
        
        # 移除控制字符（除了制表符、换行符、回车符）
        cleaned = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')
        
        # 移除BOM标记
        if cleaned.startswith('\ufeff'):
            cleaned = cleaned[1:]
        
        return cleaned
    
    def parse_json_flexible(self, text: str) -> Optional[Dict[str, Any]]:
        """灵活的JSON解析"""
        try:
            # 清理文本
            text = text.strip()
            
            # 移除代码块标记
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            # 尝试解析
            data = json.loads(text, strict=False)
            
            # 验证基本结构
            if self.validate_basic_structure(data):
                return data
            
            return None
            
        except json.JSONDecodeError:
            # 尝试修复常见的JSON问题
            try:
                # 修复尾随逗号
                text = re.sub(r',\s*}', '}', text)
                text = re.sub(r',\s*]', ']', text)
                
                # 修复Python布尔值和None
                text = re.sub(r':\s*True\b', ': true', text)
                text = re.sub(r':\s*False\b', ': false', text)
                text = re.sub(r':\s*None\b', ': null', text)
                
                data = json.loads(text, strict=False)
                
                if self.validate_basic_structure(data):
                    return data
                    
            except:
                pass
            
            return None
        except Exception:
            return None
    
    def extract_json_from_anywhere(self, text: str) -> Optional[Dict[str, Any]]:
        """从任何地方提取JSON"""
        # 查找可能的JSON对象
        patterns = [
            r'(\{[\s\S]*?"nodes"\s*:\s*\[[\s\S]*?\][\s\S]*?"edges"\s*:\s*\[[\s\S]*?\][\s\S]*?\})',
            r'(\{[\s\S]*?"edges"\s*:\s*\[[\s\S]*?\][\s\S]*?"nodes"\s*:\s*\[[\s\S]*?\][\s\S]*?\})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # 尝试修复和解析
                    cleaned = self.clean_json_text(match)
                    data = json.loads(cleaned, strict=False)
                    
                    if self.validate_basic_structure(data):
                        return data
                        
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def validate_basic_structure(self, data: Any) -> bool:
        """验证基本结构 - 简化版"""
        if not isinstance(data, dict):
            return False
        
        # 必须有nodes数组
        if 'nodes' not in data or not isinstance(data['nodes'], list):
            return False
        
        # 必须有edges数组
        if 'edges' not in data or not isinstance(data['edges'], list):
            return False
        
        # 至少需要一个节点
        if len(data['nodes']) == 0:
            return False
        
        # 验证节点基本结构
        for node in data['nodes']:
            if not isinstance(node, dict):
                return False
            if 'id' not in node:
                return False
        
        # 验证边基本结构
        for edge in data['edges']:
            if not isinstance(edge, dict):
                continue  # 边可以为空或部分无效
            if 'source' not in edge or 'target' not in edge:
                continue  # 跳过无效的边
        
        return True
    
    def clean_json_text(self, json_text: str) -> str:
        """清理JSON文本"""
        # 移除注释，避免正则在恶意输入上发生回溯
        json_text = self._strip_json_comments(json_text)
        
        # 修复尾随逗号
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # 修复Python布尔值和None
        json_text = re.sub(r':\s*True\b', ': true', json_text)
        json_text = re.sub(r':\s*False\b', ': false', json_text)
        json_text = re.sub(r':\s*None\b', ': null', json_text)
        
        return json_text

    def _strip_json_comments(self, text: str) -> str:
        """线性移除 // 和 /* */ 注释，避免正则回溯。"""
        if not text:
            return text

        result = []
        i = 0
        length = len(text)
        in_string = False
        string_quote = ''
        escaped = False

        while i < length:
            char = text[i]
            next_char = text[i + 1] if i + 1 < length else ''

            if in_string:
                result.append(char)
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == string_quote:
                    in_string = False
                    string_quote = ''
                i += 1
                continue

            if char in ('"', "'"):
                in_string = True
                string_quote = char
                result.append(char)
                i += 1
                continue

            if char == '/' and next_char == '/':
                i += 2
                while i < length and text[i] not in '\r\n':
                    i += 1
                continue

            if char == '/' and next_char == '*':
                i += 2
                while i + 1 < length and not (text[i] == '*' and text[i + 1] == '/'):
                    i += 1
                if i + 1 < length:
                    i += 2
                else:
                    i = length
                continue

            result.append(char)
            i += 1

        return ''.join(result)
    
    def save_debug_file(self, raw_text: str):
        """保存调试文件 - 仅在调试模式下启用"""
        # 安全检查: 生产环境禁用调试文件保存，避免敏感信息泄露
        if not DEBUG_MODE:
            return
        
        try:
            debug_dir = '/tmp/flowviz_debug'
            os.makedirs(debug_dir, exist_ok=True)
            
            # 注意: 调试模式下保存AI响应，可能包含敏感信息，仅用于开发调试
            file_path = os.path.join(debug_dir, f'response_{self.request_id}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            
            logger.info(f"📁 [{self.request_id}] AI响应保存到: {file_path}")
                
        except Exception as e:
            logger.error(f"❌ [{self.request_id}] 保存调试文件失败: {e}")
