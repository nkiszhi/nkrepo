# flowviz/utils/strict_validator.py
"""
Strict Validator - 简化版，更灵活的验证
"""
import re
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class StrictFlowValidator:
    """Strict Flow Validator - 简化版，更灵活的验证"""
    
    @staticmethod
    def validate_and_enhance_nodes(raw_nodes: List[Dict], source_text: str, request_id: str) -> Tuple[List[Dict], List[str], Dict]:
        """验证和增强节点 - 简化版"""
        valid_nodes = []
        warnings = []
        stats = {
            'total_attempted': len(raw_nodes),
            'validated': 0,
            'adjusted': 0,
            'rejected': 0
        }
        
        logger.info(f"🔍 [{request_id}] 验证 {len(raw_nodes)} 个节点（灵活模式）")
        
        # 为节点添加默认值
        for i, node in enumerate(raw_nodes):
            try:
                # 确保有ID
                if 'id' not in node or not node['id']:
                    node['id'] = f"node-{i}"
                    warnings.append(f"节点 {i}: 添加了默认ID")
                    stats['adjusted'] += 1
                
                # 确保有type
                if 'type' not in node:
                    node['type'] = 'default'
                    warnings.append(f"节点 {node['id']}: 添加了默认类型")
                    stats['adjusted'] += 1
                
                # 确保有data对象
                if 'data' not in node:
                    node['data'] = {}
                    warnings.append(f"节点 {node['id']}: 添加了data对象")
                    stats['adjusted'] += 1
                
                data = node['data']
                
                # 确保有label或name
                if 'label' not in data and 'name' not in data:
                    data['label'] = f"节点 {node['id']}"
                    warnings.append(f"节点 {node['id']}: 添加了默认标签")
                    stats['adjusted'] += 1
                elif 'label' not in data and 'name' in data:
                    data['label'] = data['name']
                
                # 确保有description
                if 'description' not in data:
                    if 'label' in data:
                        data['description'] = f"这是一个 {node['type']} 类型的节点：{data['label']}"
                    else:
                        data['description'] = f"这是一个 {node['type']} 类型的节点"
                    warnings.append(f"节点 {node['id']}: 添加了描述")
                    stats['adjusted'] += 1
                
                # 确保有position
                if 'position' not in node:
                    node['position'] = {'x': 0, 'y': 0}
                    warnings.append(f"节点 {node['id']}: 添加了位置")
                    stats['adjusted'] += 1
                
                valid_nodes.append(node)
                stats['validated'] += 1
                
            except Exception as e:
                logger.warning(f"⚠️ [{request_id}] 节点 {i} 验证失败: {str(e)}")
                stats['rejected'] += 1
        
        logger.info(f"📊 [{request_id}] 节点验证完成: {stats['validated']} 有效, {stats['adjusted']} 调整")
        return valid_nodes, warnings, stats
    
    @staticmethod
    def validate_edges_react_flow(edges: List[Dict], nodes: List[Dict], request_id: str) -> Tuple[List[Dict], List[str]]:
        """验证边 - 简化版"""
        valid_edges = []
        warnings = []
        
        logger.info(f"🔗 [{request_id}] 验证 {len(edges)} 条边")
        
        # 节点ID映射
        node_ids = {node['id'] for node in nodes}
        
        for i, edge in enumerate(edges):
            try:
                edge_id = edge.get('id', f'edge-{i}')
                
                # 基本验证
                if 'source' not in edge or not edge['source']:
                    warnings.append(f"边 {edge_id}: 缺少source，跳过")
                    continue
                
                if 'target' not in edge or not edge['target']:
                    warnings.append(f"边 {edge_id}: 缺少target，跳过")
                    continue
                
                source = edge['source']
                target = edge['target']
                
                # 检查源节点和目标节点是否存在
                if source not in node_ids:
                    warnings.append(f"边 {edge_id}: 源节点不存在: {source}，跳过")
                    continue
                
                if target not in node_ids:
                    warnings.append(f"边 {edge_id}: 目标节点不存在: {target}，跳过")
                    continue
                
                # 确保有ID
                if 'id' not in edge or not edge['id']:
                    edge['id'] = f"edge-{len(valid_edges)}"
                
                # 确保有标签
                if 'label' not in edge:
                    edge['label'] = '相关'
                    warnings.append(f"边 {edge['id']}: 添加了默认标签")
                
                # 确保有type
                if 'type' not in edge:
                    edge['type'] = 'default'
                
                valid_edges.append(edge)
                
            except Exception as e:
                logger.warning(f"⚠️ [{request_id}] 边 {i} 验证失败: {str(e)}")
                continue
        
        logger.info(f"✅ [{request_id}] 边验证: {len(valid_edges)} 有效, {len(warnings)} 警告")
        return valid_edges, warnings
    
    @staticmethod
    def convert_to_react_flow_compatible(nodes: List[Dict], edges: List[Dict], request_id: str) -> Dict:
        """转换为React Flow兼容格式"""
        logger.info(f"🔄 [{request_id}] 转换为React Flow格式")
        
        # 重新布局节点位置
        nodes = StrictFlowValidator._arrange_nodes_in_grid(nodes)
        
        # 构建结果
        result = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'exportedAt': datetime.utcnow().isoformat() + 'Z',
                'tool': 'FlowViz',
                'version': '1.0.0',
                'nodeCount': len(nodes),
                'edgeCount': len(edges),
                'layoutDirection': 'TB',
                'format': 'react_flow',
                'requestId': request_id
            }
        }
        
        logger.info(f"✅ [{request_id}] 转换完成: {len(nodes)} 个节点, {len(edges)} 条边")
        return result
    
    @staticmethod
    def _arrange_nodes_in_grid(nodes: List[Dict]) -> List[Dict]:
        """按网格排列节点位置"""
        if not nodes:
            return nodes
        
        # 网格布局
        cols = 3
        node_width = 250
        node_height = 180
        padding_x = 100
        padding_y = 100
        
        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            
            node['position'] = {
                'x': padding_x + col * node_width,
                'y': padding_y + row * node_height
            }
        
        return nodes