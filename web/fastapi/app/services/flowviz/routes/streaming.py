# flowviz/routes/streaming.py
#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
FlowViz 流式处理核心 - 实时解析AI响应
"""
import json
import re
import logging
import time
import traceback
from datetime import datetime
from flask import Blueprint, request, Response, jsonify, stream_with_context
from ..providers.factory import ProviderFactory

logger = logging.getLogger(__name__)
bp = Blueprint('streaming', __name__)

class StreamingFlowParser:
    """实时流式解析器 - 类似原始FlowViz项目的逻辑"""
    
    def __init__(self, request_id):
        self.request_id = request_id
        self.response_text = ""
        self.processed_node_ids = set()
        self.processed_edge_ids = set()
        self.pending_edges = []
        self.emitted_node_ids = set()
        
    def process_chunk(self, chunk_text):
        """处理一个文本块，尝试提取节点和边"""
        if not chunk_text:
            return [], []
        
        self.response_text += chunk_text
        
        nodes = self._extract_nodes()
        edges = self._extract_edges()
        
        return nodes, edges
    
    def _extract_nodes(self):
        """从响应文本中提取节点"""
        nodes = []
        
        # 匹配节点JSON对象 - 使用非贪婪匹配避免ReDoS
        # 原正则: r'\{\s*["\']?id["\']?\s*:\s*["\']([^"\']+)["\'][^}]*["\']?type["\']?\s*:\s*["\']([^"\']+)["\'][^}]*\}'
        # 优化: 使用非贪婪匹配[^}]*?替代贪婪匹配[^}]*，避免指数级回溯
        node_pattern = r'\{\s*["\']?id["\']?\s*:\s*["\']([^"\']+)["\'][^}]*?["\']?type["\']?\s*:\s*["\']([^"\']+)["\'][^}]*?\}'
        
        matches = list(re.finditer(node_pattern, self.response_text, re.DOTALL))
        
        for match in matches:
            node_str = match.group(0)
            node_id = match.group(1)
            
            if node_id not in self.processed_node_ids:
                try:
                    # 尝试解析为JSON
                    node_data = json.loads(node_str)
                    
                    # 标准化节点格式
                    if 'id' in node_data and 'type' in node_data:
                        self.processed_node_ids.add(node_id)
                        nodes.append(node_data)
                        
                except json.JSONDecodeError:
                    # 可能是不完整的JSON，尝试修复
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
                                        # 找到完整的JSON对象
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
        
        # 匹配边JSON对象 - 使用非贪婪匹配避免ReDoS
        # 原正则: r'\{\s*["\']?id["\']?\s*:\s*["\']([^"\']+)["\'][^}]*["\']?source["\']?\s*:\s*["\']([^"\']+)["\'][^}]*["\']?target["\']?\s*:\s*["\']([^"\']+)["\'][^}]*\}'
        # 优化: 使用非贪婪匹配[^}]*?替代贪婪匹配[^}]*，避免指数级回溯
        edge_pattern = r'\{\s*["\']?id["\']?\s*:\s*["\']([^"\']+)["\'][^}]*?["\']?source["\']?\s*:\s*["\']([^"\']+)["\'][^}]*?["\']?target["\']?\s*:\s*["\']([^"\']+)["\'][^}]*?\}'
        
        matches = list(re.finditer(edge_pattern, self.response_text, re.DOTALL))
        
        for match in matches:
            edge_str = match.group(0)
            edge_id = match.group(1)
            source_id = match.group(2)
            target_id = match.group(3)
            
            if edge_id not in self.processed_edge_ids:
                try:
                    edge_data = json.loads(edge_str)
                    
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
    
    def mark_node_emitted(self, node_id):
        """标记节点已发出"""
        self.emitted_node_ids.add(node_id)

def sse_message(event_type, data):
    """生成SSE格式消息"""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"

def sse_done():
    """生成SSE完成消息"""
    return "data: [DONE]\n\n"

@bp.route('/analyze-stream-realtime', methods=['POST'])
def analyze_stream_realtime():
    """实时流式分析端点 - 类似原始FlowViz项目"""
    start_time = datetime.now()
    request_id = f"stream_{int(time.time())}_{hash(str(start_time)) % 10000}"
    
    logger.info(f"🚀 [{request_id}] 开始实时流式分析")
    
    try:
        # 解析请求数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体为空'}), 400
        
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4o')  # 默认使用 gpt-4o
        url = data.get('url')
        text = data.get('text')
        system = data.get('system', "你是网络威胁情报分析方面的专家。请严格按照要求的JSON格式返回分析结果。")
        
        # 获取分析内容
        content = ""
        if url:
            # 获取URL内容（简化的实现）
            content = f"URL内容: {url}"
        elif text:
            content = text
        else:
            return jsonify({'error': '必须提供 url 或 text 参数'}), 400
        
        # 获取提供商配置
        provider_config = ProviderFactory.get_provider_config(provider)
        if not provider_config:
            return jsonify({'error': f'不支持的AI提供商: {provider}'}), 400
        
        # 更新模型配置
        provider_config['model'] = model
        
        # 创建解析器
        parser = StreamingFlowParser(request_id)
        
        def generate():
            """实时流式生成器"""
            try:
                # 发送初始化进度
                yield sse_message('progress', {
                    'stage': 'initializing',
                    'message': '正在初始化分析引擎...',
                    'percentage': 10
                })
                
                time.sleep(0.1)
                
                # 创建AI提供商
                yield sse_message('progress', {
                    'stage': 'creating_provider',
                    'message': f'正在创建{provider}提供商连接...',
                    'percentage': 20
                })
                
                ai_provider = ProviderFactory.create(provider, provider_config)
                # 注意: 只记录提供商名称，不记录包含api_key的provider_config
                logger.info(f"[{request_id}] AI提供商创建成功: {ai_provider.get_name()}")
                
                time.sleep(0.1)
                
                # 开始AI分析
                yield sse_message('progress', {
                    'stage': 'analyzing',
                    'message': '正在使用AI分析攻击流程...',
                    'percentage': 30
                })
                
                # 构建分析参数
                params = {
                    'text': content[:20000],  # 限制长度
                    'system': system,
                    'model': model
                }
                
                # 收集AI响应
                response_text = ""
                
                # 调用AI提供商的stream方法
                for chunk in ai_provider.stream(params, None):
                    if chunk:
                        # 如果是数据块，尝试提取内容
                        if chunk.startswith('data: ') and chunk != 'data: [DONE]\n\n':
                            try:
                                chunk_data = json.loads(chunk[6:].strip())
                                if chunk_data.get('type') == 'content_block_delta':
                                    delta_text = chunk_data.get('delta', {}).get('text', '')
                                    if delta_text:
                                        response_text += delta_text
                                        
                                        # 实时解析
                                        nodes, edges = parser.process_chunk(delta_text)
                                        
                                        # 发送节点
                                        for node in nodes:
                                            parser.mark_node_emitted(node['id'])
                                            yield sse_message('node', {'node': node})
                                        
                                        # 发送边
                                        for edge in edges:
                                            yield sse_message('edge', {'edge': edge})
                                        
                                        # 发送进度更新
                                        total_parsed = len(parser.processed_node_ids) + len(parser.processed_edge_ids)
                                        progress = min(80, 30 + int(total_parsed * 50 / 20))  # 假设最多20个元素
                                        
                                        yield sse_message('progress', {
                                            'stage': 'parsing',
                                            'message': f'已解析 {len(parser.processed_node_ids)} 个节点, {len(parser.processed_edge_ids)} 条边',
                                            'percentage': progress
                                        })
                                        
                            except json.JSONDecodeError:
                                continue
                        
                        # 发送原始块到前端（用于调试）
                        yield chunk
                
                # 解析完成后的剩余内容
                yield sse_message('progress', {
                    'stage': 'finalizing',
                    'message': '正在完成分析...',
                    'percentage': 90
                })
                
                time.sleep(0.1)
                
                # 发送完成信号
                yield sse_message('progress', {
                    'stage': 'complete',
                    'message': '分析完成！',
                    'percentage': 100
                })
                
                yield sse_done()
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ [{request_id}] 实时流式分析完成，总耗时: {duration:.2f}秒")
                
            except Exception as e:
                logger.error(f"❌ [{request_id}] 流式生成器错误: {str(e)}")
                yield sse_message('error', {'error': f'处理错误: {str(e)}'})
                yield sse_done()
        
        # 返回SSE响应
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"❌ [{request_id}] 分析请求处理失败: {str(e)}")
        # 注意: 不向用户暴露详细的错误信息
        return jsonify({'error': '服务器内部错误，请稍后重试'}), 500

# 添加健康检查路由
@bp.route('/health', methods=['GET'])
def streaming_health():
    """流式模块健康检查"""
    return jsonify({
        'status': 'healthy',
        'module': 'streaming',
        'endpoints': {
            'analyze-stream-realtime': '/flowviz/api/analyze-stream-realtime',
            'health': '/flowviz/api/health'
        }
    })

# 添加测试路由
@bp.route('/test', methods=['GET'])
def streaming_test():
    """流式模块测试"""
    return jsonify({
        'success': True,
        'message': 'Streaming module is working',
        'endpoint': '/flowviz/api/streaming'
    })