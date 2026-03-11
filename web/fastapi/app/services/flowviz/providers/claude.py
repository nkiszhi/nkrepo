#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
Claude Provider Implementation - 简化版
"""
import anthropic
import json
import logging
import time
import traceback
from .base import BaseProvider

logger = logging.getLogger(__name__)

class ClaudeProvider(BaseProvider):
    """Claude Provider - 简化版"""
    
    def __init__(self, config):
        super().__init__(config)
        self.provider_name = 'Claude'
        self.client = None
        
        logger.info(f"[Claude] 初始化客户端")
        logger.info(f"[Claude] API URL: {self.base_url}")
        logger.info(f"[Claude] 模型: {self.model}")
        
        try:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"[Claude] 客户端初始化成功")
        except Exception as e:
            logger.error(f"[Claude] 客户端初始化失败: {e}")
            raise
    
    def stream(self, params, response_generator):
        """流式分析"""
        text = params.get('text')
        vision_analysis = params.get('visionAnalysis')
        system = params.get('system')
        model = params.get('model', self.model)
        
        input_length = len(text) if text else 0
        logger.info(f"[Claude] 输入文本长度: {input_length} 字符")
        
        max_input_chars = 100000
        if text and len(text) > max_input_chars:
            logger.warning(f"[Claude] 警告: 文本过长 ({len(text)} 字符), 截断到 {max_input_chars}")
            text = text[:max_input_chars]
        
        # 构建消息
        messages = self.format_prompt(text, vision_analysis, system)
        
        try:
            logger.info(f"[Claude] 开始流式分析")
            
            with self.client.messages.stream(
                model=model,
                max_tokens=4096,
                temperature=0.2,
                system=messages[0]['content'],
                messages=messages[1:]
            ) as stream:
                for chunk in stream.text_stream:
                    if chunk:
                        yield f"data: {json.dumps({'type': 'content_block_delta', 'delta': {'text': chunk}})}\n\n"
            
            yield "data: [DONE]\n\n"
            logger.info(f"[Claude] 流式分析完成")
            
        except Exception as error:
            logger.error(f"[Claude] 流式分析错误: {str(error)}")
            logger.error(traceback.format_exc())
            
            error_msg = str(error)
            error_detail = ""
            
            if 'context_length' in error_msg.lower():
                error_detail = "上下文长度超出限制。请使用更短的文本。"
            elif 'quota' in error_msg.lower():
                error_detail = "API配额已用完。"
            elif 'rate_limit' in error_msg.lower():
                error_detail = "API频率限制。"
            elif 'invalid_api_key' in error_msg.lower():
                error_detail = "无效的API密钥。"
            else:
                error_detail = f"API调用失败: {error_msg[:200]}"
            
            yield f"data: {json.dumps({'type': 'error', 'error': error_detail})}\n\n"
            yield "data: [DONE]\n\n"
    
    def format_prompt(self, text, vision_analysis, system):
        """格式化提示词 - 简化版"""
        final_text = text or ""
        
        if vision_analysis:
            final_text = f"## 图像分析结果\n\n{vision_analysis}\n\n## 文章文本\n\n{text}"
        
        # 简化提示词
        user_prompt = f"""你是一名网络安全威胁情报分析师。请分析以下网络安全报告，并根据内容创建一个攻击流程图。

分析报告内容：
"{final_text[:80000]}"

任务要求：
1. 将报告中的攻击步骤、技术、工具、基础设施等元素提取出来
2. 创建攻击流程图，展示攻击的各个阶段和它们之间的关系
3. 使用React Flow兼容的JSON格式返回结果

输出格式要求：
请返回一个JSON对象，包含两个数组：
1. "nodes": 包含图中所有节点的数组
2. "edges": 包含所有节点之间边的数组

节点格式示例：
{{
  "id": "node-1",
  "type": "action",
  "position": {{"x": 100, "y": 100}},
  "data": {{
    "label": "鱼叉式钓鱼邮件",
    "description": "攻击者发送带有恶意附件的钓鱼邮件"
  }}
}}

边格式示例：
{{
  "id": "edge-1",
  "source": "node-1",
  "target": "node-2",
  "label": "导致"
}}

请根据报告内容创建准确、逻辑清晰的攻击流程图。
只返回JSON对象，不要有其他文字。"""
        
        system_prompt = system or """你是网络安全威胁情报分析专家，擅长从安全报告中提取攻击信息并创建可视化流程图。"""
        
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    
    def analyze_vision(self, images, article_text, prompt=None):
        """分析图像"""
        # Claude可能需要不同的视觉分析实现
        # 这里简化处理
        logger.info(f"[Claude] 视觉分析暂不支持")
        return {
            'analysisText': f"图像分析: {len(images)} 张图片，文章长度: {len(article_text)} 字符",
            'tokensUsed': 0
        }
    
    @staticmethod
    def get_supported_models():
        """获取支持的模型列表"""
        return [
            'claude-3-5-sonnet-20241022',
            'claude-3-5-sonnet-20240620',
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307'
        ]
    
    def is_configured(self):
        """验证配置"""
        return bool(self.api_key and self.model)