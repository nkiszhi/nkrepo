#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
OpenAI GPT Provider Implementation - FlowViz严格模式版本
优化位置生成算法，支持动态画布布局
"""
import openai
import json
import logging
import time
import traceback
import re
from datetime import datetime
from .base import BaseProvider
from ..config import FlowVizConfig

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseProvider):
    """OpenAI GPT Provider - FlowViz严格模式版本"""
    
    MODEL_LIMITS = {
        'gpt-4o': {'max_tokens': 16384, 'context_window': 128000},
        'gpt-4o-mini': {'max_tokens': 16384, 'context_window': 128000},
        'gpt-4-turbo': {'max_tokens': 4096, 'context_window': 128000},
        'gpt-4': {'max_tokens': 8192, 'context_window': 8192},
        'gpt-3.5-turbo': {'max_tokens': 4096, 'context_window': 16385},
        'gpt-3.5-turbo-16k': {'max_tokens': 16384, 'context_window': 16384},
    }
    
    def __init__(self, config):
        super().__init__(config)
        self.provider_name = 'OpenAI'
        self.api_version = self.detect_api_version()
        self.total_tokens_used = 0
        self.response_start_time = None
        
        # FlowViz严格模式设置
        self.strict_mode = config.get('strict_mode', FlowVizConfig.FLOW_STRICT_MODE)
        self.required_technique_ids = config.get('required_technique_ids', FlowVizConfig.REQUIRED_TECHNIQUE_IDS)
        self.require_mitre_mapping = config.get('require_mitre_mapping', FlowVizConfig.REQUIRE_MITRE_MAPPING)
        
        # 稳定性控制参数
        self.temperature = 0.1  # 低温度以获得更一致的结果
        self.top_p = 0.9
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.1
        
        # 布局参数
        self.node_width = 280  # 节点宽度
        self.node_height = 160  # 节点高度
        self.node_spacing_x = 350  # 节点水平间距
        self.node_spacing_y = 250  # 节点垂直间距
        
        logger.info(f"[OpenAI] 初始化客户端")
        # 注意: 不记录包含敏感信息的配置，只记录必要的调试信息
        logger.info(f"[OpenAI] 模型: {self.model}")
        logger.info(f"[OpenAI] 严格模式: {self.strict_mode}")
        logger.info(f"[OpenAI] 需要技术ID: {self.required_technique_ids}")
        
        try:
            if self.api_version == '0.28.1':
                openai.api_key = self.api_key
                openai.api_base = self.base_url
                logger.info(f"[OpenAI] 使用 0.28.1 版本客户端")
            else:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info(f"[OpenAI] 使用 1.x 版本客户端")
            
            logger.info(f"[OpenAI] 客户端初始化成功")
        except Exception as e:
            logger.error(f"[OpenAI] 客户端初始化失败: {e}")
            raise
    
    def detect_api_version(self):
        """检测OpenAI版本"""
        try:
            version = openai.__version__
            logger.info(f"[OpenAI] 检测到版本: {version}")
            if version.startswith('0.'):
                return '0.28.1'
            else:
                return '1.x'
        except:
            return 'unknown'
    
    def stream(self, params, response_generator):
        """流式分析 - FlowViz严格模式版本"""
        text = params.get('text')
        vision_analysis = params.get('visionAnalysis')
        system = params.get('system')
        model = params.get('model', self.model)
        strict_mode = params.get('strict_mode', self.strict_mode)
        
        input_length = len(text) if text else 0
        logger.info(f"[OpenAI] 输入文本长度: {input_length} 字符")
        
        model_limit = self.MODEL_LIMITS.get(model, {'max_tokens': 4096, 'context_window': 4096})
        max_completion_tokens = model_limit['max_tokens']
        context_window = model_limit['context_window']
        
        logger.info(f"[OpenAI] 模型限制: max_tokens={max_completion_tokens}, context_window={context_window}")
        
        estimated_input_tokens = input_length // 2
        available_output_tokens = context_window - estimated_input_tokens - 1000
        
        safe_max_tokens = min(max_completion_tokens, available_output_tokens, 16384)
        
        if input_length > 30000:
            safe_max_tokens = min(16384, safe_max_tokens)
        elif input_length > 15000:
            safe_max_tokens = min(12288, safe_max_tokens)
        elif input_length > 8000:
            safe_max_tokens = min(8192, safe_max_tokens)
        else:
            safe_max_tokens = min(4096, safe_max_tokens)
        
        logger.info(f"[OpenAI] 安全 max_tokens: {safe_max_tokens}")
        
        max_input_chars = 50000
        if text and len(text) > max_input_chars:
            logger.warning(f"[OpenAI] 警告: 文本过长 ({len(text)} 字符), 截断到 {max_input_chars}")
            text = text[:max_input_chars]
        
        messages = self.format_strict_prompt(text, vision_analysis, system, strict_mode, safe_max_tokens)
        
        try:
            # 注意: 只记录模型参数，不记录包含api_key的配置信息
            logger.info(f"[OpenAI] 开始流式分析")
            logger.info(f"[OpenAI] 模型: {model}, max_tokens: {safe_max_tokens}, strict_mode: {strict_mode}")
            
            self.response_start_time = time.time()
            
            temperature = self.temperature
            top_p = self.top_p
            frequency_penalty = self.frequency_penalty
            presence_penalty = self.presence_penalty
            
            if self.api_version == '0.28.1':
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=safe_max_tokens,
                    stream=True,
                )
                
                full_response = ""
                chunk_count = 0
                for chunk in response:
                    chunk_count += 1
                    if 'choices' in chunk and len(chunk.choices) > 0:
                        delta = chunk.choices[0].get('delta', {})
                        content = delta.get('content', '')
                        
                        if content:
                            full_response += content
                            yield f"data: {json.dumps({'type': 'content_block_delta', 'delta': {'text': content}})}\n\n"
                
                logger.info(f"[OpenAI] 完整响应长度: {len(full_response)} 字符, 块数: {chunk_count}")
                
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=safe_max_tokens,
                    stream=True,
                )
                
                full_response = ""
                chunk_count = 0
                for chunk in response:
                    chunk_count += 1
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content = delta.content
                            full_response += content
                            yield f"data: {json.dumps({'type': 'content_block_delta', 'delta': {'text': content}})}\n\n"
                
                logger.info(f"[OpenAI] 完整响应长度: {len(full_response)} 字符, 块数: {chunk_count}")
            
            response_duration = time.time() - self.response_start_time
            logger.info(f"[OpenAI] 响应时间: {response_duration:.2f} 秒")
            
            yield "data: [DONE]\n\n"
            logger.info(f"[OpenAI] 流式分析完成")
            
        except Exception as error:
            logger.error(f"[OpenAI] 流式分析错误: {str(error)}")
            logger.error(traceback.format_exc())
            
            error_msg = str(error)
            error_detail = ""
            
            if 'max_tokens is too large' in error_msg:
                error_detail = "模型输出长度限制为16384个token。请简化输入或分段分析。"
            elif 'context_length' in error_msg.lower():
                error_detail = "上下文长度超出限制。请使用更短的文本或升级模型。"
            elif 'quota' in error_msg.lower():
                error_detail = "API配额已用完。"
            elif 'rate_limit' in error_msg.lower():
                error_detail = "API频率限制。"
            elif 'model_not_found' in error_msg.lower():
                error_detail = f"模型 {model} 未找到。"
            elif 'invalid_api_key' in error_msg.lower():
                error_detail = "无效的API密钥。"
            elif 'timeout' in error_msg.lower():
                error_detail = "API调用超时。请检查网络连接。"
            else:
                error_detail = f"API调用失败: {error_msg[:200]}"
            
            yield f"data: {json.dumps({'type': 'error', 'error': error_detail})}\n\n"
            yield "data: [DONE]\n\n"
    
    def format_strict_prompt(self, text, vision_analysis, system, strict_mode, max_tokens):
        """生成FlowViz严格模式提示词 - 优化位置生成算法"""
        final_text = text or ""
        
        if vision_analysis:
            final_text = f"## 图像分析结果\n\n{vision_analysis}\n\n## 文章文本\n\n{text}"
        
        # MITRE ATT&CK技术分类参考
        mitre_tactics = """
MITRE ATT&CK战术分类:
1. Initial Access (初始访问): T1190, T1566, T1189, T1195, T1192, T1193, T1194, T1199, T1200, T1201, T1078
2. Execution (执行): T1059, T1203, T1129, T1204, T1569, T1053, T1106, T1127, T1220, T1559, T1056
3. Persistence (持久化): T1133, T1136, T1547, T1037, T1176, T1554, T1137, T1197, T1505, T1525, T1543, T1574
4. Privilege Escalation (权限提升): T1068, T1134, T1548, T1484, T1611, T1055, T1078, T1547
5. Defense Evasion (防御规避): T1140, T1197, T1211, T1222, T1562, T1574, T1070, T1112, T1202, T1027, T1553
6. Credential Access (凭证访问): T1110, T1555, T1556, T1557, T1558, T1606, T1056, T1111, T1187, T1212, T1528
7. Discovery (发现): T1016, T1018, T1021, T1033, T1046, T1049, T1057, T1069, T1082, T1083, T1087, T1120, T1135
8. Lateral Movement (横向移动): T1021, T1072, T1080, T1091, T1135, T1210, T1534, T1550, T1563, T1570
9. Collection (收集): T1113, T1114, T1115, T1119, T1123, T1125, T1185, T1213, T1530, T1557, T1560
10. Exfiltration (数据外泄): T1020, T1030, T1041, T1048, T1052, T1567, T1537, T1029, T1011
11. Command and Control (命令与控制): T1071, T1095, T1102, T1104, T1105, T1132, T1219, T1571, T1572, T1573
"""
        
        # 改进的位置生成算法示例
        position_examples = """
动态位置生成算法:
- 根据攻击阶段（战术）分组节点
- 同一战术的节点垂直对齐
- 不同战术的节点水平排列
- 保持合理的间距避免重叠
示例位置:
- Initial Access阶段: x: 100-500, y: 100-300
- Execution阶段: x: 600-1000, y: 100-300
- Persistence阶段: x: 100-500, y: 400-600
- 确保节点位置在合理范围内 (x: 50-3000, y: 50-2000)
"""
        
        # 严格的提示词 - 基于原版FlowViz格式
        user_prompt = f"""You are an expert in cyber threat intelligence and MITRE ATT&CK framework. Analyze this cybersecurity article and create a professional attack flow diagram.

IMPORTANT: You MUST return ONLY a valid JSON object with "nodes" and "edges" arrays. NO additional text before or after.

{mitre_tactics}

{position_examples}

VALID NODE TYPES:
- action: MITRE ATT&CK technique nodes (MUST include valid technique_id)
- tool: Software/tools used in attack (e.g., Metasploit, Cobalt Strike)
- malware: Malicious software (e.g., Emotet, TrickBot)
- asset: Target systems or resources (e.g., Web Server, Database)
- infrastructure: C2 servers, domains, IP addresses
- url: Web resources or phishing URLs
- vulnerability: CVEs or specific vulnerabilities exploited
- AND_operator: Logical AND gate (attack requires multiple conditions)
- OR_operator: Logical OR gate (attack can follow multiple paths)

REQUIRED NODE FIELDS:
1. id: Unique identifier (e.g., "action-1", "tool-1")
2. type: One of the valid node types above
3. position: {{"x": number, "y": number}} (use smart grid layout based on attack phase)
4. data: Object containing:
   - name: Clear descriptive name
   - description: Detailed explanation
   - confidence: "low", "medium", or "high"
   - source_excerpt: Relevant quote from the article (if applicable)
   - For action nodes ONLY: technique_id (e.g., "T1190"), tactic (e.g., "Initial Access"), platform, data_sources
   - For technical nodes: ip, domain, hash, command, file_path, registry_key as applicable

VALID EDGE TYPES/LABELS:
- "uses": Tool/technique uses another resource
- "targets": Attack targets specific asset
- "communicates_with": Network communication
- "exploits": Exploits vulnerability
- "creates": Creates file/process/artifact
- "modifies": Modifies configuration/data
- "leads_to": Leads to next attack phase
- "affects": Affects system/component

{"="*60}
STRICT REQUIREMENTS (ENABLED):
1. ALL action nodes MUST have valid MITRE ATT&CK technique IDs (format T####)
2. Action nodes MUST include: technique_id, tactic, platform (e.g., "Windows", "Linux")
3. Organize nodes by attack phase (tactics) in chronological order
4. Use operator nodes (AND/OR) for complex attack logic
5. Validate all technical indicators against known patterns
6. Include specific technical details: IPs, domains, hashes, commands
7. Ensure logical flow with clear progression
{"="*60 if strict_mode else ""}

POSITION GENERATION GUIDELINES:
1. Start with x: 100, y: 100
2. Group nodes by attack tactic:
   - Initial Access: x: 100-500, y: 100-300
   - Execution: x: 100-500, y: 400-600
   - Persistence: x: 600-1000, y: 100-300
   - Privilege Escalation: x: 600-1000, y: 400-600
   - Defense Evasion: x: 1100-1500, y: 100-300
   - Credential Access: x: 1100-1500, y: 400-600
3. Space nodes appropriately: x spacing 350, y spacing 250
4. Ensure no overlapping nodes
5. Keep all positions within reasonable bounds: x: 50-3000, y: 50-2000
6. For large diagrams, extend to x: 50-5000, y: 50-3000

OUTPUT FORMAT EXAMPLE:
{{
  "nodes": [
    {{
      "id": "action-1",
      "type": "action",
      "position": {{"x": 100, "y": 100}},
      "data": {{
        "name": "Spearphishing Attachment",
        "description": "Attacker sends malicious attachment via email",
        "technique_id": "T1566.001",
        "tactic": "Initial Access",
        "platform": "Windows",
        "data_sources": ["Network Traffic", "Email Gateway"],
        "confidence": "high",
        "source_excerpt": "The attack began with a phishing email containing a malicious Word document."
      }}
    }},
    {{
      "id": "malware-1",
      "type": "malware",
      "position": {{"x": 400, "y": 100}},
      "data": {{
        "name": "Emotet",
        "description": "Banking Trojan used for initial access",
        "hash": "a1b2c3d4e5f678901234567890123456",
        "confidence": "high",
        "source_excerpt": "The attachment contained the Emotet banking Trojan."
      }}
    }},
    {{
      "id": "action-2",
      "type": "action",
      "position": {{"x": 100, "y": 400}},
      "data": {{
        "name": "PowerShell Execution",
        "description": "Executes PowerShell script to download payload",
        "technique_id": "T1059.001",
        "tactic": "Execution",
        "platform": "Windows",
        "data_sources": ["Process Monitoring", "Command Line"],
        "confidence": "high",
        "source_excerpt": "The malware executed PowerShell to download additional payloads."
      }}
    }}
  ],
  "edges": [
    {{
      "id": "edge-1",
      "source": "action-1",
      "target": "malware-1",
      "label": "delivers",
      "type": "floating"
    }},
    {{
      "id": "edge-2",
      "source": "malware-1",
      "target": "action-2",
      "label": "executes",
      "type": "floating"
    }}
  ]
}}

ARTICLE TEXT TO ANALYZE:
"{final_text[:40000]}"

Important: Generate positions that make sense for the attack flow. Group related nodes together.
Keep x coordinates between 50 and 3000, y coordinates between 50 and 2000.
Use consistent spacing to create a readable diagram.

Return ONLY the JSON object:"""
        
        system_prompt = system or """You are an expert cyber threat intelligence analyst specialized in MITRE ATT&CK framework.
Your task is to analyze cybersecurity articles and extract accurate attack patterns.
You must return valid JSON only, with no additional text.
Follow ALL requirements strictly, especially for MITRE ATT&CK technique mapping.
Generate positions that create a logical, readable attack flow diagram."""
        
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    
    # 兼容性方法
    def format_prompt(self, text, vision_analysis, system):
        return self.format_strict_prompt(text, vision_analysis, system, self.strict_mode, 4096)
    
    def analyze_vision(self, images, article_text, prompt=None):
        """分析图像"""
        if not images or not isinstance(images, list) or len(images) == 0:
            raise Exception('缺少或无效的图像数组')
        
        if self.api_version == '0.28.1':
            logger.error(f"[OpenAI] 0.28.1 版本不支持Vision API")
            raise Exception("当前OpenAI版本不支持Vision API")
        
        try:
            vision_prompt = prompt or self.build_vision_prompt(article_text, len(images))
            
            content = self.build_message_content(images, vision_prompt)
            
            logger.info(f"[OpenAI] 开始视觉分析，图像数量: {len(images)}")
            
            vision_model = self.get_vision_model()
            
            response = self.client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
            
            logger.info(f"[OpenAI] 视觉分析完成，token使用: {tokens_used}")
            
            return {
                'analysisText': analysis_text,
                'tokensUsed': tokens_used
            }
            
        except Exception as error:
            logger.error(f"[OpenAI] 视觉分析错误: {str(error)}")
            raise
    
    def build_vision_prompt(self, article_text, image_count):
        """构建视觉分析提示词"""
        return f"""You are analyzing {image_count} images from a cybersecurity article to enhance threat intelligence analysis.

Article context (first 1000 chars):
{article_text[:1000]}...

Please analyze the images and provide:
1. Technical details visible in screenshots (commands, file paths, network indicators)
2. Attack techniques or tools shown
3. Any MITRE ATT&CK relevant information (technique IDs, tactics)
4. System configurations or vulnerabilities displayed
5. Technical indicators of compromise (IOCs)

Focus on actionable technical intelligence that supplements the article text."""
    
    def build_message_content(self, images, prompt):
        """构建消息内容（包含图像）"""
        content = [{"type": "text", "text": prompt}]
        
        for image in images:
            if image.get('base64') and image.get('mediaType'):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image['mediaType']};base64,{image['base64']}"
                    }
                })
        
        return content
    
    def get_vision_model(self):
        """获取支持视觉的模型"""
        vision_models = ['gpt-4o', 'gpt-4o-2024-11-20', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4-vision-preview']
        return self.model if self.model in vision_models else 'gpt-4o'
    
    @staticmethod
    def get_supported_models():
        """获取支持的模型列表"""
        return [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4-turbo',
            'gpt-4',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k',
        ]
    
    def is_configured(self):
        """验证配置"""
        return bool(self.api_key and self.model)
    
    def get_token_usage(self):
        """获取token使用量"""
        return self.total_tokens_used
    
    def get_provider_info(self):
        """获取提供商信息"""
        return {
            'name': 'OpenAI',
            'version': self.api_version,
            'strict_mode': self.strict_mode,
            'required_technique_ids': self.required_technique_ids,
            'supports_vision': self.api_version != '0.28.1',
            'model': self.model,
            'token_usage': self.total_tokens_used,
            'prompt_style': 'flowviz_strict',
            'node_width': self.node_width,
            'node_height': self.node_height,
            'node_spacing': {
                'x': self.node_spacing_x,
                'y': self.node_spacing_y
            }
        }