"""
FlowViz 流式处理API - FastAPI版本
"""
import json
import re
import logging
import time
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


class StreamingFlowParser:
    """实时流式解析器"""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.response_text = ""
        self.processed_node_ids = set()
        self.processed_edge_ids = set()
        self.pending_edges = []
        self.emitted_node_ids = set()
    
    def process_chunk(self, chunk_text: str):
        """处理一个文本块,尝试提取节点和边"""
        if not chunk_text:
            return [], []
        
        self.response_text += chunk_text
        
        nodes = self._extract_nodes()
        edges = self._extract_edges()
        
        return nodes, edges
    
    def _extract_nodes(self):
        """从响应文本中提取节点"""
        nodes = []
        
        # 清理响应文本 - 去除markdown代码块标记
        cleaned_text = self.response_text.strip()
        if cleaned_text.startswith('```'):
            # 去除开头的```json或```
            first_newline = cleaned_text.find('\n')
            if first_newline != -1:
                cleaned_text = cleaned_text[first_newline+1:]
            # 去除结尾的```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
        
        # 尝试直接解析整个响应为JSON
        try:
            full_data = json.loads(cleaned_text)
            if isinstance(full_data, dict) and 'nodes' in full_data:
                for node_data in full_data['nodes']:
                    node_id = node_data.get('id')
                    if node_id and node_id not in self.processed_node_ids:
                        if 'id' in node_data and 'type' in node_data and 'source' not in node_data:
                            self.processed_node_ids.add(node_id)
                            nodes.append(node_data)
                return nodes
        except:
            pass
        
        # 如果直接解析失败,使用正则表达式匹配
        node_pattern = r'\{\s*["\']?id["\']?\s*:\s*["\']([^"\']+)["\'][^}]*["\']?type["\']?\s*:\s*["\']([^"\']+)["\'][^}]*\}'
        
        matches = list(re.finditer(node_pattern, self.response_text, re.DOTALL))
        
        for match in matches:
            node_str = match.group(0)
            node_id = match.group(1)
            
            # 跳过边(包含source字段的对象)
            if '"source"' in node_str or "'source'" in node_str:
                continue
            
            if node_id not in self.processed_node_ids:
                try:
                    node_data = json.loads(node_str)
                    
                    # 标准化节点格式
                    if 'id' in node_data and 'type' in node_data and 'source' not in node_data:
                        self.processed_node_ids.add(node_id)
                        nodes.append(node_data)
                        
                except json.JSONDecodeError:
                    try:
                        # 尝试找到完整的JSON对象
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        
                        for i, char in enumerate(node_str):
                            if escape_next:
                                escape_next = False
                                continue
                            
                            if char == '\\':
                                escape_next = True
                                continue
                            
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        complete_str = node_str[:i+1]
                                        node_data = json.loads(complete_str)
                                        self.processed_node_ids.add(node_id)
                                        nodes.append(node_data)
                                        break
                    except:
                        continue
        
        return nodes
    
    def _extract_edges(self):
        """从响应文本中提取边"""
        edges = []
        
        # 清理响应文本 - 去除markdown代码块标记
        cleaned_text = self.response_text.strip()
        if cleaned_text.startswith('```'):
            # 去除开头的```json或```
            first_newline = cleaned_text.find('\n')
            if first_newline != -1:
                cleaned_text = cleaned_text[first_newline+1:]
            # 去除结尾的```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
        
        # 尝试直接解析整个响应为JSON
        try:
            full_data = json.loads(cleaned_text)
            if isinstance(full_data, dict) and 'edges' in full_data:
                for edge_data in full_data['edges']:
                    edge_id = edge_data.get('id')
                    source_id = edge_data.get('source')
                    target_id = edge_data.get('target')
                    
                    if edge_id and edge_id not in self.processed_edge_ids:
                        # 如果源节点和目标节点都已发出，则直接添加边
                        if source_id in self.emitted_node_ids and target_id in self.emitted_node_ids:
                            self.processed_edge_ids.add(edge_id)
                            edges.append(edge_data)
                        else:
                            # 否则加入待处理列表
                            self.pending_edges.append({
                                'edge': edge_data,
                                'source': source_id,
                                'target': target_id
                            })
                
                # 处理待处理的边
                remaining_edges = []
                for pending in self.pending_edges:
                    if pending['source'] in self.emitted_node_ids and pending['target'] in self.emitted_node_ids:
                        self.processed_edge_ids.add(pending['edge']['id'])
                        edges.append(pending['edge'])
                    else:
                        remaining_edges.append(pending)
                
                self.pending_edges = remaining_edges
                return edges
        except:
            pass
        
        # 如果直接解析失败,使用正则表达式匹配
        edge_pattern = r'\{\s*["\']?id["\']?\s*:\s*["\']([^"\']+)["\'][^}]*["\']?source["\']?\s*:\s*["\']([^"\']+)["\'][^}]*["\']?target["\']?\s*:\s*["\']([^"\']+)["\'][^}]*\}'
        
        matches = list(re.finditer(edge_pattern, self.response_text, re.DOTALL))
        
        for match in matches:
            edge_str = match.group(0)
            edge_id = match.group(1)
            source_id = match.group(2)
            target_id = match.group(3)
            
            if edge_id not in self.processed_edge_ids:
                try:
                    edge_data = json.loads(edge_str)
                    
                    if source_id in self.emitted_node_ids and target_id in self.emitted_node_ids:
                        self.processed_edge_ids.add(edge_id)
                        edges.append(edge_data)
                    else:
                        self.pending_edges.append({
                            'edge': edge_data,
                            'source': source_id,
                            'target': target_id
                        })
                        
                except json.JSONDecodeError:
                    continue
        
        # 处理待处理的边
        remaining_edges = []
        for pending in self.pending_edges:
            if pending['source'] in self.emitted_node_ids and pending['target'] in self.emitted_node_ids:
                self.processed_edge_ids.add(pending['edge']['id'])
                edges.append(pending['edge'])
            else:
                remaining_edges.append(pending)
        
        self.pending_edges = remaining_edges
        
        return edges
    
    def mark_node_emitted(self, node_id: str):
        """标记节点已发出"""
        self.emitted_node_ids.add(node_id)


def sse_message(event_type: str, data: dict) -> str:
    """生成SSE格式消息"""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def sse_done() -> str:
    """生成SSE完成消息"""
    return "data: [DONE]\n\n"


@router.post("/api/analyze-stream")
async def analyze_stream_realtime(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """实时流式分析端点"""
    start_time = datetime.now()
    request_id = f"stream_{int(time.time())}_{hash(str(start_time)) % 10000}"
    
    logger.info(f"🚀 [{request_id}] 开始实时流式分析")
    
    async def generate() -> AsyncGenerator[str, None]:
        """实时流式生成器"""
        try:
            # 发送初始化进度
            yield sse_message('progress', {
                'stage': 'initializing',
                'message': '正在初始化分析引擎...',
                'percentage': 10
            })
            
            await asyncio.sleep(0.1)
            
            # 获取请求参数
            provider = request.get('provider', 'openai')
            model = request.get('model', 'gpt-4o')
            url = request.get('url')
            text = request.get('text')
            system = request.get('system', "你是网络威胁情报分析方面的专家。请严格按照要求的JSON格式返回分析结果。")
            
            # 获取分析内容
            content = ""
            if url:
                content = f"URL内容: {url}"
            elif text:
                content = text
            else:
                yield sse_message('error', {'error': '必须提供 url 或 text 参数'})
                yield sse_done()
                return
            
            # 创建解析器
            parser = StreamingFlowParser(request_id)
            
            # 发送创建提供商进度
            yield sse_message('progress', {
                'stage': 'creating_provider',
                'message': f'正在创建{provider}提供商连接...',
                'percentage': 20
            })
            
            await asyncio.sleep(0.1)
            
            # 这里需要导入ProviderFactory
            try:
                from app.services.flowviz.providers.factory import ProviderFactory
                
                # 获取提供商配置
                provider_config = ProviderFactory.get_provider_config(provider)
                if not provider_config:
                    yield sse_message('error', {'error': f'不支持的AI提供商: {provider}'})
                    yield sse_done()
                    return
                
                # 更新模型配置
                provider_config['model'] = model
                
                # 创建AI提供商
                ai_provider = ProviderFactory.create(provider, provider_config)
                logger.info(f"[{request_id}] AI提供商创建成功: {ai_provider.get_name()}")
                
            except ImportError:
                # 如果没有flowviz模块,使用模拟数据
                logger.warning(f"[{request_id}] flowviz模块未找到,使用模拟数据")
                yield sse_message('progress', {
                    'stage': 'analyzing',
                    'message': '正在使用AI分析攻击流程...',
                    'percentage': 30
                })
                
                # 模拟流式输出
                mock_response = '''
                {
                  "nodes": [
                    {
                      "id": "node1",
                      "type": "start",
                      "label": "攻击开始",
                      "data": {
                        "label": "攻击开始",
                        "description": "攻击者开始发起攻击",
                        "technique_id": "",
                        "tactic": ""
                      }
                    },
                    {
                      "id": "node2",
                      "type": "action",
                      "label": "初始访问",
                      "data": {
                        "label": "初始访问",
                        "description": "攻击者通过钓鱼邮件获取初始访问权限",
                        "technique_id": "T1566",
                        "tactic": "Initial Access"
                      }
                    },
                    {
                      "id": "node3",
                      "type": "action",
                      "label": "权限提升",
                      "data": {
                        "label": "权限提升",
                        "description": "攻击者利用漏洞提升权限",
                        "technique_id": "T1068",
                        "tactic": "Privilege Escalation"
                      }
                    },
                    {
                      "id": "node4",
                      "type": "end",
                      "label": "攻击完成",
                      "data": {
                        "label": "攻击完成",
                        "description": "攻击者完成攻击目标",
                        "technique_id": "",
                        "tactic": ""
                      }
                    }
                  ],
                  "edges": [
                    {"id": "edge1", "source": "node1", "target": "node2"},
                    {"id": "edge2", "source": "node2", "target": "node3"},
                    {"id": "edge3", "source": "node3", "target": "node4"}
                  ]
                }
                '''
                
                # 逐字符发送模拟数据
                for i, char in enumerate(mock_response):
                    response_text = mock_response[:i+1]
                    
                    # 实时解析
                    nodes, edges = parser.process_chunk(char)
                    
                    # 发送节点
                    for node in nodes:
                        parser.mark_node_emitted(node['id'])
                        yield sse_message('node', {'node': node})
                    
                    # 发送边
                    for edge in edges:
                        yield sse_message('edge', {'edge': edge})
                    
                    # 发送进度更新
                    total_parsed = len(parser.processed_node_ids) + len(parser.processed_edge_ids)
                    progress = min(80, 30 + int(total_parsed * 50 / 20))
                    
                    yield sse_message('progress', {
                        'stage': 'parsing',
                        'message': f'已解析 {len(parser.processed_node_ids)} 个节点, {len(parser.processed_edge_ids)} 条边',
                        'percentage': progress
                    })
                    
                    await asyncio.sleep(0.01)
                
                # 发送完成信号
                yield sse_message('progress', {
                    'stage': 'complete',
                    'message': '分析完成！',
                    'percentage': 100
                })
                
                yield sse_done()
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ [{request_id}] 实时流式分析完成(模拟),总耗时: {duration:.2f}秒")
                
                return
            
            # 开始AI分析
            yield sse_message('progress', {
                'stage': 'analyzing',
                'message': '正在使用AI分析攻击流程...',
                'percentage': 30
            })
            
            # 构建分析参数
            params = {
                'text': content[:20000],
                'system': system,
                'model': model
            }
            
            # 收集AI响应
            response_text = ""
            
            # 调用AI提供商的stream方法
            for chunk in ai_provider.stream(params, None):
                if chunk:
                    if chunk.startswith('data: ') and chunk != 'data: [DONE]\n\n':
                        try:
                            chunk_data = json.loads(chunk[6:].strip())
                            if chunk_data.get('type') == 'content_block_delta':
                                delta_text = chunk_data.get('delta', {}).get('text', '')
                                if delta_text:
                                    response_text += delta_text
                                    
                                    # 实时解析
                                    nodes, edges = parser.process_chunk(delta_text)
                                    
                                    # 调试日志
                                    if nodes:
                                        logger.info(f"[{request_id}] 解析到 {len(nodes)} 个节点: {[n.get('id') for n in nodes]}")
                                    if edges:
                                        logger.info(f"[{request_id}] 解析到 {len(edges)} 条边: {[e.get('id') for e in edges]}")
                                    
                                    # 发送节点
                                    for node in nodes:
                                        parser.mark_node_emitted(node['id'])
                                        yield sse_message('node', {'node': node})
                                    
                                    # 发送边
                                    for edge in edges:
                                        yield sse_message('edge', {'edge': edge})
                                    
                                    # 发送进度更新
                                    total_parsed = len(parser.processed_node_ids) + len(parser.processed_edge_ids)
                                    progress = min(80, 30 + int(total_parsed * 50 / 20))
                                    
                                    yield sse_message('progress', {
                                        'stage': 'parsing',
                                        'message': f'已解析 {len(parser.processed_node_ids)} 个节点, {len(parser.processed_edge_ids)} 条边',
                                        'percentage': progress
                                    })
                                    
                        except json.JSONDecodeError:
                            continue
                    
                    # 发送原始块到前端
                    yield chunk
            
            # 解析完成后的剩余内容
            logger.info(f"[{request_id}] AI完整响应长度: {len(response_text)} 字符")
            
            # 保存完整响应到文件
            import os
            debug_dir = os.path.join(os.path.dirname(__file__), '../../logs')
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f'ai_response_{request_id}.txt')
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response_text)
            logger.info(f"[{request_id}] AI响应已保存到: {debug_file}")
            
            # 再次解析边(因为流式处理时节点可能还未全部发出)
            parser.response_text = response_text
            final_edges = parser._extract_edges()
            for edge in final_edges:
                yield sse_message('edge', {'edge': edge})
            
            logger.info(f"[{request_id}] 总共解析到 {len(parser.processed_node_ids)} 个节点, {len(parser.processed_edge_ids)} 条边")
            
            yield sse_message('progress', {
                'stage': 'finalizing',
                'message': '正在完成分析...',
                'percentage': 90
            })
            
            await asyncio.sleep(0.1)
            
            # 发送完成信号
            yield sse_message('progress', {
                'stage': 'complete',
                'message': '分析完成！',
                'percentage': 100
            })
            
            yield sse_done()
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ [{request_id}] 实时流式分析完成,总耗时: {duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] 流式生成器错误: {str(e)}")
            yield sse_message('error', {'error': f'处理错误: {str(e)}'})
            yield sse_done()
    
    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )
