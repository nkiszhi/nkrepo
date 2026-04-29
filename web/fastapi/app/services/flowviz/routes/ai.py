#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
FlowViz AI Routes Module - 严格模式版本
"""
from flask import Blueprint, request, Response, jsonify, stream_with_context
import json
import logging
import re
import traceback
import time
import os
import hashlib
import threading
from datetime import datetime, timedelta
from ..providers.factory import ProviderFactory
from ..utils.advanced_parser import AdvancedFlowParser
from ..utils.technical_processor import TechnicalDataProcessor
from ..utils.sse import sse_message, sse_done
from ..config import FlowVizConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('ai', __name__)

# 严格模式验证器
class StrictFlowValidator:
    """严格模式验证器 - 类似原版FlowViz"""
    
    def __init__(self, request_id):
        self.request_id = request_id
    
    def validate_strict_compliance(self, nodes, edges, strict_mode):
        """验证严格模式合规性"""
        if not strict_mode:
            return nodes, edges, []
        
        warnings = []
        validated_nodes = []
        
        # 验证每个节点
        for node in nodes:
            node_type = node.get('type', '')
            data = node.get('data', {})
            
            # 检查action节点是否有技术ID
            if node_type == 'action':
                if 'technique_id' not in data or not data['technique_id']:
                    warnings.append(f"Action节点 {node.get('id')} 缺少MITRE ATT&CK技术ID")
                    continue
                
                # 验证技术ID格式
                tech_id = data.get('technique_id', '')
                if not re.match(r'^T\d{4}(\.\d{3})?$', tech_id):
                    warnings.append(f"Action节点 {node.get('id')} 的技术ID格式无效: {tech_id}")
                
                # 检查是否缺少战术信息
                if 'tactic' not in data or not data['tactic']:
                    warnings.append(f"Action节点 {node.get('id')} 缺少战术信息")
            
            # 检查节点是否有必要的字段
            if 'id' not in node:
                node['id'] = f"node-{len(validated_nodes)}"
            
            if 'type' not in node:
                node['type'] = 'default'
                warnings.append(f"节点 {node['id']} 缺少类型，设为默认")
            
            if 'data' not in node:
                node['data'] = {}
                warnings.append(f"节点 {node['id']} 缺少data字段")
            
            # 确保data中有必要字段
            data = node['data']
            if 'name' not in data or not data['name']:
                if 'label' in data:
                    data['name'] = data['label']
                else:
                    data['name'] = f"{node['type']} {node['id']}"
                    warnings.append(f"节点 {node['id']} 缺少名称")
            
            if 'description' not in data:
                data['description'] = f"{data.get('name', '节点')}的描述"
                warnings.append(f"节点 {node['id']} 缺少描述")
            
            if 'confidence' not in data:
                data['confidence'] = 'medium'
            
            validated_nodes.append(node)
        
        # 验证边
        validated_edges = []
        node_ids = {node['id'] for node in validated_nodes}
        
        for i, edge in enumerate(edges):
            if 'id' not in edge:
                edge['id'] = f"edge-{i}"
            
            if 'source' not in edge or 'target' not in edge:
                warnings.append(f"边 {edge['id']} 缺少源节点或目标节点")
                continue
            
            if edge['source'] not in node_ids:
                warnings.append(f"边 {edge['id']} 的源节点不存在: {edge['source']}")
                continue
            
            if edge['target'] not in node_ids:
                warnings.append(f"边 {edge['id']} 的目标节点不存在: {edge['target']}")
                continue
            
            if 'label' not in edge or not edge['label']:
                edge['label'] = '相关'
            
            if 'type' not in edge:
                edge['type'] = 'floating'
            
            validated_edges.append(edge)
        
        # 统计信息
        action_nodes = [n for n in validated_nodes if n.get('type') == 'action']
        action_with_tech = [n for n in action_nodes if n.get('data', {}).get('technique_id')]
        
        stats = {
            'total_nodes': len(validated_nodes),
            'action_nodes': len(action_nodes),
            'action_with_tech': len(action_with_tech),
            'edges': len(validated_edges),
            'warnings': len(warnings),
            'compliance_rate': len(action_with_tech) / len(action_nodes) if action_nodes else 1.0
        }
        
        logger.info(f"✅ [{self.request_id}] 严格模式验证完成: {stats}")
        
        return validated_nodes, validated_edges, warnings, stats
    
    def enhance_with_mitre_info(self, nodes):
        """增强MITRE ATT&CK信息"""
        enhanced_nodes = []
        
        # MITRE ATT&CK技术参考
        mitre_techniques = {
            'T1190': {'name': 'Exploit Public-Facing Application', 'tactic': 'Initial Access'},
            'T1566': {'name': 'Phishing', 'tactic': 'Initial Access'},
            'T1566.001': {'name': 'Spearphishing Attachment', 'tactic': 'Initial Access'},
            'T1059': {'name': 'Command and Scripting Interpreter', 'tactic': 'Execution'},
            'T1059.003': {'name': 'Windows Command Shell', 'tactic': 'Execution'},
            'T1547': {'name': 'Boot or Logon Autostart Execution', 'tactic': 'Persistence'},
            'T1068': {'name': 'Exploitation for Privilege Escalation', 'tactic': 'Privilege Escalation'},
            'T1027': {'name': 'Obfuscated Files or Information', 'tactic': 'Defense Evasion'},
            'T1110': {'name': 'Brute Force', 'tactic': 'Credential Access'},
            'T1082': {'name': 'System Information Discovery', 'tactic': 'Discovery'},
            'T1021': {'name': 'Remote Services', 'tactic': 'Lateral Movement'},
            'T1113': {'name': 'Screen Capture', 'tactic': 'Collection'},
            'T1048': {'name': 'Exfiltration Over Alternative Protocol', 'tactic': 'Exfiltration'},
            'T1071': {'name': 'Application Layer Protocol', 'tactic': 'Command and Control'},
        }
        
        for node in nodes:
            if node.get('type') == 'action':
                data = node.get('data', {})
                tech_id = data.get('technique_id', '')
                
                if tech_id in mitre_techniques:
                    mitre_info = mitre_techniques[tech_id]
                    if 'name' not in data or not data['name']:
                        data['name'] = mitre_info['name']
                    if 'tactic' not in data or not data['tactic']:
                        data['tactic'] = mitre_info['tactic']
                    
                    # 添加MITRE链接
                    data['mitre_url'] = f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}/"
            
            enhanced_nodes.append(node)
        
        return enhanced_nodes

# 缓存分析结果
class AnalysisCache:
    """分析结果缓存"""
    
    def __init__(self, max_size=100, ttl_hours=24):
        self.cache = {}
        self.lock = threading.Lock()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
    
    def _generate_cache_key(self, provider, model, content, strict_mode):
        """生成缓存键"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{provider}:{model}:{strict_mode}:{content_hash}"
    
    def _clean_expired(self):
        """清理过期缓存"""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if now - entry['timestamp'] >= self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            
        if expired_keys:
            logger.info(f"🗑️ 清理了 {len(expired_keys)} 个过期缓存项")
    
    def get(self, provider, model, content, strict_mode):
        """从缓存获取结果"""
        with self.lock:
            cache_key = self._generate_cache_key(provider, model, content, strict_mode)
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if datetime.now() - entry['timestamp'] < self.ttl:
                    logger.info(f"📦 缓存命中: {cache_key}")
                    return entry['result']
                else:
                    del self.cache[cache_key]
            return None
    
    def set(self, provider, model, content, strict_mode, result):
        """设置缓存结果"""
        with self.lock:
            cache_key = self._generate_cache_key(provider, model, content, strict_mode)
            
            # 清理过期缓存
            self._clean_expired()
            
            # 如果缓存已满，删除最旧的项
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]
                logger.info(f"🗑️ 缓存已满，删除最旧项: {oldest_key}")
            
            self.cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now(),
                'provider': provider,
                'model': model,
                'strict_mode': strict_mode,
                'content_length': len(content)
            }
            logger.info(f"💾 缓存结果: {cache_key}")

# 创建全局缓存实例
analysis_cache = AnalysisCache(max_size=100, ttl_hours=24)

@bp.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    """
    AI流式分析端点 - 严格模式版本
    """
    start_time = datetime.now()
    request_id = f"req_{int(time.time())}_{hash(str(start_time)) % 10000}"
    
    logger.info("=" * 80)
    logger.info(f"📥 [{request_id}] 收到流式分析请求")
    
    try:
        if not request.data:
            logger.error(f"❌ [{request_id}] 请求体为空")
            return jsonify({'error': '请求体为空'}), 400
        
        raw_data_bytes = request.get_data()
        logger.info(f"📋 [{request_id}] 收到原始字节，长度: {len(raw_data_bytes)}")
        
        raw_data = raw_data_bytes.decode('utf-8', errors='replace')
        logger.info(f"📋 [{request_id}] 解码后数据长度: {len(raw_data)}")
        
        if not raw_data or raw_data.strip() == '':
            logger.error(f"❌ [{request_id}] 请求体为空")
            return jsonify({'error': '请求体为空'}), 400
        
        try:
            data = json.loads(raw_data)
            logger.info(f"✅ [{request_id}] JSON解析成功，字段: {list(data.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ [{request_id}] JSON解析失败: {str(e)}")
            return jsonify({'error': '无法解析JSON请求体'}), 400
        
        provider = data.get('provider', '')
        model = data.get('model', '')
        url = data.get('url', '')
        text = data.get('text', '')
        strict_mode = data.get('strict_mode', FlowVizConfig.FLOW_STRICT_MODE)
        system = data.get('system', "你是网络威胁情报分析专家，擅长MITRE ATT&CK框架。根据报告内容创建准确的攻击流程图。")
        
        logger.info(f"📋 [{request_id}] 请求参数:")
        logger.info(f"  - 提供商: {provider}")
        logger.info(f"  - 模型: {model}")
        logger.info(f"  - 严格模式: {strict_mode}")
        logger.info(f"  - URL: {url if url else '无'}")
        logger.info(f"  - 文本长度: {len(text) if text else 0}")
        
        # 确定使用哪个提供商
        if not provider:
            provider = ProviderFactory.get_default_provider()
            logger.info(f"🔄 [{request_id}] 使用默认提供商: {provider}")
        
        if not provider:
            logger.error(f"❌ [{request_id}] 没有配置AI提供商")
            return jsonify({'error': '没有配置AI提供商。请配置OPENAI_API_KEY或CLAUDE_API_KEY。'}), 500
        
        # 如果没有模型，使用默认模型
        if not model:
            provider_config = ProviderFactory.get_provider_config(provider)
            model = provider_config.get('model', 'gpt-4o' if provider == 'openai' else 'claude-3-5-sonnet-20241022')
            # 注意: 只记录模型名称，不记录包含api_key的provider_config
            logger.info(f"🔄 [{request_id}] 使用默认模型: {model}")
        
        content = ""
        if url:
            logger.info(f"🌐 [{request_id}] 处理URL: {url}")
            try:
                from .fetch import secure_fetch, SimpleReadability
                
                response = secure_fetch(url, {
                    'timeout': 30,
                    'headers': {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
                })
                
                if response.ok:
                    html = response.text
                    result = SimpleReadability.extract(html, url)
                    content = result.get('content', html)
                    logger.info(f"✅ [{request_id}] URL内容获取成功，长度: {len(content)}")
                else:
                    logger.error(f"❌ [{request_id}] 获取URL失败: {response.status_code}")
                    return jsonify({'error': f'无法获取URL内容: {response.reason}'}), 400
                    
            except ImportError:
                logger.warning(f"⚠️ [{request_id}] fetch模块不可用，使用URL作为文本")
                content = f"URL: {url}\n\n请分析此URL的网络安全内容。"
            except Exception as e:
                logger.error(f"❌ [{request_id}] URL处理错误: {str(e)}")
                content = f"URL: {url}\n\n错误: {str(e)}"
                
        elif text:
            logger.info(f"📝 [{request_id}] 处理文本输入，原始长度: {len(text)}")
            content = preprocess_text_input(text, request_id)
            logger.info(f"✅ [{request_id}] 文本预处理完成，长度: {len(content)}")
            
        else:
            logger.error(f"❌ [{request_id}] 未提供URL或文本参数")
            return jsonify({'error': '必须提供url或text参数'}), 400
        
        if not content or len(content.strip()) == 0:
            logger.error(f"❌ [{request_id}] 处理后的内容为空")
            return jsonify({'error': '无法获取有效的分析内容'}), 400
        
        max_content_length = 50000
        if len(content) > max_content_length:
            logger.warning(f"⚠️ [{request_id}] 内容过长，截断到前 {max_content_length} 字符")
            content = content[:max_content_length]
        
        logger.info(f"✅ [{request_id}] 最终分析内容长度: {len(content)} 字符")
        
        # 检查缓存
        cached_result = analysis_cache.get(provider, model, content, strict_mode)
        
        if cached_result:
            logger.info(f"🎯 [{request_id}] 缓存命中！使用缓存的分析结果")
            return generate_cached_response(cached_result, request_id, strict_mode)
        
        # 保存原始内容用于调试
        try:
            debug_dir = '/tmp/flowviz_debug'
            os.makedirs(debug_dir, exist_ok=True)
            content_path = os.path.join(debug_dir, f'final_content_{request_id}.txt')
            with open(content_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"📁 [{request_id}] 最终内容保存到: {content_path}")
        except Exception as e:
            logger.warning(f"⚠️ [{request_id}] 保存调试文件失败: {e}")
        
        logger.info(f"🔧 [{request_id}] 获取提供商配置: {provider}")
        provider_config = ProviderFactory.get_provider_config(provider)
        
        if not provider_config:
            logger.error(f"❌ [{request_id}] 不支持的提供商: {provider}")
            return jsonify({'error': f'不支持的AI提供商: {provider}'}), 400
        
        api_key = provider_config.get('api_key')
        if not api_key:
            # 注意: 只记录错误信息，不记录api_key本身
            logger.error(f"❌ [{request_id}] {provider} API密钥未配置")
            return jsonify({'error': f'{provider} API密钥未配置，请检查配置文件'}), 500
        
        # 注意: 只记录配置状态，不记录包含api_key的provider_config
        logger.info(f"✅ [{request_id}] 提供商配置有效")
        logger.info(f"🚀 [{request_id}] 开始生成流响应...")
        
        def generate():
            """流响应生成器 - 严格模式版本"""
            nodes = []
            edges = []
            response_collected = ""
            
            try:
                # 进度初始化
                yield sse_message('progress', {
                    'stage': 'initializing', 
                    'message': '初始化FlowViz分析引擎...', 
                    'percentage': 5,
                    'status': '',
                    'strict_mode': strict_mode
                })
                
                time.sleep(0.1)
                
                # 创建AI提供商
                yield sse_message('progress', {
                    'stage': 'creating_provider', 
                    'message': f'创建 {provider} 提供商连接...', 
                    'percentage': 15,
                    'status': ''
                })
                
                try:
                    provider_config['model'] = model
                    provider_config['strict_mode'] = strict_mode
                    # 注意: 只记录模型和模式，不记录包含api_key的provider_config
                    logger.info(f"🔄 [{request_id}] 使用模型: {model}, 严格模式: {strict_mode}")
                    
                    ai_provider = ProviderFactory.create(provider, provider_config)
                    # 注意: 只记录提供商名称，不记录包含api_key的provider_config
                    logger.info(f"✅ [{request_id}] AI提供商创建成功: {ai_provider.get_name()}")
                except Exception as e:
                    logger.error(f"❌ [{request_id}] 创建AI提供商失败: {str(e)}")
                    yield sse_message('error', {'error': f'创建AI提供商失败: {str(e)}'})
                    yield sse_done()
                    return
                
                time.sleep(0.1)
                
                # 开始AI分析
                yield sse_message('progress', {
                    'stage': 'analyzing', 
                    'message': '使用AI分析攻击流程...', 
                    'percentage': 30,
                    'status': ''
                })
                
                params = {
                    'text': content,
                    'system': system,
                    'model': model,
                    'strict_mode': strict_mode
                }
                
                logger.info(f"📤 [{request_id}] 调用AI提供商流式方法，内容长度: {len(content)}")
                chunk_count = 0
                
                try:
                    for chunk in ai_provider.stream(params, None):
                        chunk_count += 1
                        
                        if chunk:
                            yield chunk
                            
                            if chunk.startswith('data: ') and chunk != 'data: [DONE]\n\n':
                                try:
                                    chunk_data = json.loads(chunk[6:].strip())
                                    
                                    if chunk_data.get('type') == 'content_block_delta':
                                        delta_text = chunk_data.get('delta', {}).get('text', '')
                                        if delta_text:
                                            response_collected += delta_text
                                            
                                            if len(response_collected) % 2000 == 0:
                                                logger.info(f"📝 [{request_id}] 已收集响应长度: {len(response_collected)} 字符")
                                    
                                except json.JSONDecodeError:
                                    if chunk.startswith('data: '):
                                        try:
                                            text_content = chunk[6:].strip()
                                            if text_content and text_content != '[DONE]':
                                                response_collected += text_content
                                        except:
                                            pass
                    
                    logger.info(f"✅ [{request_id}] AI分析完成，处理 {chunk_count} 个数据块")
                    logger.info(f"📄 [{request_id}] 完整AI响应长度: {len(response_collected)} 字符")
                    
                except Exception as e:
                    logger.error(f"❌ [{request_id}] AI分析错误: {str(e)}")
                    yield sse_message('error', {'error': f'AI分析失败: {str(e)}'})
                    yield sse_done()
                    return
                
                time.sleep(0.1)
                
                # 解析AI响应
                yield sse_message('progress', {
                    'stage': 'parsing', 
                    'message': '解析AI响应...', 
                    'percentage': 70,
                    'status': ''
                })
                
                if response_collected and len(response_collected) > 100:
                    logger.info(f"📄 [{request_id}] 开始解析，长度: {len(response_collected)}")
                    
                    # 使用高级解析器
                    parser = AdvancedFlowParser(request_id)
                    parsed_result = parser.parse_technical_data(response_collected)
                    
                    if 'error' in parsed_result:
                        logger.error(f"❌ [{request_id}] 解析AI响应失败: {parsed_result['error']}")
                        yield sse_message('error', {'error': parsed_result['error']})
                        
                        # 发送原始响应用于调试
                        yield sse_message('raw_response', {
                            'text': response_collected[:2000],
                            'full_length': len(response_collected)
                        })
                    else:
                        nodes = parsed_result.get('nodes', [])
                        edges = parsed_result.get('edges', [])
                        
                        logger.info(f"✅ [{request_id}] 解析成功: {len(nodes)} 个节点, {len(edges)} 条边")
                        
                        # 严格模式验证
                        yield sse_message('progress', {
                            'stage': 'validating', 
                            'message': '验证严格模式合规性...', 
                            'percentage': 80,
                            'status': ''
                        })
                        
                        validator = StrictFlowValidator(request_id)
                        validated_nodes, validated_edges, warnings, stats = validator.validate_strict_compliance(
                            nodes, edges, strict_mode
                        )
                        
                        # 增强MITRE信息
                        enhanced_nodes = validator.enhance_with_mitre_info(validated_nodes)
                        
                        # 缓存结果
                        cache_result = {
                            'nodes': enhanced_nodes,
                            'edges': validated_edges,
                            'stats': stats,
                            'warnings': warnings,
                            'response_length': len(response_collected),
                            'strict_mode': strict_mode
                        }
                        analysis_cache.set(provider, model, content, strict_mode, cache_result)
                        logger.info(f"💾 [{request_id}] 分析结果已缓存")
                        
                        # 发送节点
                        for node in enhanced_nodes:
                            yield sse_message('node', {
                                'node': node,
                                'format': 'flowviz_strict',
                                'strict_mode': strict_mode
                            })
                        
                        # 发送边
                        for edge in validated_edges:
                            yield sse_message('edge', {
                                'edge': edge,
                                'format': 'flowviz_strict'
                            })
                        
                        # 发送统计信息
                        yield sse_message('stats', {
                            'nodes': len(enhanced_nodes),
                            'edges': len(validated_edges),
                            'action_nodes': stats.get('action_nodes', 0),
                            'action_with_tech': stats.get('action_with_tech', 0),
                            'compliance_rate': stats.get('compliance_rate', 0),
                            'warnings': len(warnings),
                            'format': 'flowviz_strict',
                            'strict_mode': strict_mode,
                            'response_length': len(response_collected)
                        })
                        
                        # 发送警告（如果有）
                        if warnings:
                            yield sse_message('warnings', {
                                'warnings': warnings[:10],  # 只发送前10个警告
                                'total_warnings': len(warnings)
                            })
                else:
                    logger.warning(f"⚠️ [{request_id}] AI响应过短: {len(response_collected) if response_collected else 0} 字符")
                    yield sse_message('error', {'error': 'AI未返回足够的分析内容'})
                    yield sse_message('raw_response', {
                        'text': response_collected[:2000] if response_collected else '',
                        'full_length': len(response_collected) if response_collected else 0
                    })
                
                # 完成
                yield sse_message('progress', {
                    'stage': 'complete', 
                    'message': '分析完成！攻击流程生成成功。', 
                    'percentage': 100,
                    'status': 'success'
                })
                yield sse_done()
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ [{request_id}] 流式分析完成，总时间: {duration:.2f} 秒")
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"❌ [{request_id}] 流生成器内部错误: {str(e)}")
                logger.error(traceback.format_exc())
                yield sse_message('error', {'error': f'内部处理错误: {str(e)}'})
                yield sse_done()
        
        logger.info(f"📨 [{request_id}] 返回SSE流响应")
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            }
        )
        
    except Exception as e:
        logger.error(f"❌ [{request_id}] 分析请求处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        # 注意: 不向用户暴露详细的错误信息
        return jsonify({'error': '服务器内部错误，请稍后重试'}), 500

def generate_cached_response(cached_result, request_id, strict_mode):
    """从缓存生成响应"""
    def generate():
        try:
            yield sse_message('progress', {
                'stage': 'initializing', 
                'message': '加载缓存分析...', 
                'percentage': 10,
                'status': '',
                'strict_mode': strict_mode
            })
            
            time.sleep(0.1)
            
            yield sse_message('progress', {
                'stage': 'cache_hit', 
                'message': '缓存命中！使用先前分析的结果...', 
                'percentage': 30,
                'status': ''
            })
            
            nodes = cached_result.get('nodes', [])
            edges = cached_result.get('edges', [])
            stats = cached_result.get('stats', {})
            warnings = cached_result.get('warnings', [])
            
            logger.info(f"🎯 [{request_id}] 从缓存生成: {len(nodes)} 个节点, {len(edges)} 条边")
            
            # 发送节点
            for node in nodes:
                yield sse_message('node', {
                    'node': node,
                    'format': 'flowviz_strict',
                    'strict_mode': strict_mode
                })
            
            # 发送边
            for edge in edges:
                yield sse_message('edge', {
                    'edge': edge,
                    'format': 'flowviz_strict'
                })
            
            # 发送统计信息
            yield sse_message('stats', {
                'nodes': len(nodes),
                'edges': len(edges),
                'action_nodes': stats.get('action_nodes', 0),
                'action_with_tech': stats.get('action_with_tech', 0),
                'compliance_rate': stats.get('compliance_rate', 0),
                'warnings': len(warnings),
                'format': 'flowviz_strict',
                'strict_mode': strict_mode,
                'cached': True,
                'response_length': cached_result.get('response_length', 0)
            })
            
            # 发送警告（如果有）
            if warnings:
                yield sse_message('warnings', {
                    'warnings': warnings[:10],
                    'total_warnings': len(warnings)
                })
            
            yield sse_message('progress', {
                'stage': 'complete', 
                'message': '缓存分析加载成功！', 
                'percentage': 100,
                'status': 'success'
            })
            
            yield sse_done()
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] 缓存响应生成错误: {str(e)}")
            yield sse_message('error', {'error': f'缓存响应错误: {str(e)}'})
            yield sse_done()
    
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

def preprocess_text_input(text, request_id):
    """预处理文本输入"""
    logger.info(f"🔍 [{request_id}] 预处理文本输入，原始长度: {len(text)}")
    
    try:
        # 尝试解析JSON
        data = json.loads(text)
        logger.info(f"✅ [{request_id}] 输入是JSON格式")
        
        # 如果是JSON，格式化为易于阅读的文本
        formatted_text = json.dumps(data, indent=2, ensure_ascii=False)
        
        final_text = f"""网络安全分析数据：

{formatted_text}

请根据以上数据创建攻击流程图，使用MITRE ATT&CK框架。"""
        
        logger.info(f"✅ [{request_id}] 预处理完成，最终长度: {len(final_text)}")
        return final_text
        
    except json.JSONDecodeError:
        logger.info(f"📝 [{request_id}] 输入不是JSON，保持原始文本")
        return text
    except Exception as e:
        logger.warning(f"⚠️ [{request_id}] 文本预处理意外错误: {str(e)}")
        return text