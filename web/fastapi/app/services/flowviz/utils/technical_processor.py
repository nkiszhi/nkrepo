# flowviz/utils/technical_processor.py
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TechnicalDataProcessor:
    """Technical Data Processor - 简化版：只做最小化预处理，保持原始数据完整"""
    
    @staticmethod
    def preprocess_technical_data(input_data: str) -> str:
        """预处理技术数据：只是格式化JSON，不提取内容"""
        logger.info(f"🔄 预处理技术数据，长度: {len(input_data)}")
        
        try:
            # 尝试解析JSON，只是为了验证格式
            data = json.loads(input_data)
            
            # 如果是嵌套的data结构，提取出来
            if isinstance(data, dict) and 'data' in data:
                technical_data = data['data']
                logger.info(f"📊 提取嵌套data结构")
                
                # 格式化JSON以便AI阅读
                formatted_json = json.dumps(technical_data, indent=2, ensure_ascii=False)
                
                # 构建AI友好的文本描述
                ai_ready_text = f"""以下是网络安全分析报告的技术数据（JSON格式）：

{formatted_json}

请基于这些技术数据创建MITRE ATT&CK攻击流程图。"""
                
                return ai_ready_text
            else:
                # 如果不是嵌套结构，直接返回原始JSON
                formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
                
                ai_ready_text = f"""以下是网络安全分析报告的技术数据（JSON格式）：

{formatted_json}

请基于这些技术数据创建MITRE ATT&CK攻击流程图。"""
                
                return ai_ready_text
                
        except json.JSONDecodeError:
            logger.info("⚠️ 输入不是JSON格式，直接作为文本传递")
            return input_data
        except Exception as e:
            logger.error(f"❌ JSON解析失败: {e}")
            return input_data
    
    @staticmethod
    def convert_json_to_ai_text(json_data: Dict[str, Any]) -> str:
        """将JSON数据转换为AI友好的文本格式（简单描述）"""
        try:
            # 创建简单的文本描述
            lines = ["网络安全分析报告技术数据汇总："]
            
            # 列出关键数据字段
            for key, value in json_data.items():
                if isinstance(value, list):
                    lines.append(f"- {key}: {len(value)}个条目")
                elif isinstance(value, dict):
                    lines.append(f"- {key}: {len(value)}个字段")
                else:
                    lines.append(f"- {key}: {str(value)[:50]}...")
            
            lines.append("\n完整JSON数据请见下文。")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"JSON转文本失败: {e}")
            return "网络安全技术数据（详见JSON格式）"