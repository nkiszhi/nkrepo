import json
from abc import ABC, abstractmethod

class BaseProvider(ABC):
    """AI服务基础提供商类"""
    
    def __init__(self, config):
        self.api_key = config.get('api_key')
        self.model = config.get('model')
        self.base_url = config.get('base_url')
        self.provider_name = config.get('provider_name', 'unknown')
    
    @abstractmethod
    def stream(self, params, response_generator):
        """流式分析文章内容（同步版本）"""
        raise NotImplementedError(f"stream() must be implemented by {self.provider_name} provider")
    
    @abstractmethod
    def analyze_vision(self, image_data, media_type, prompt):
        """分析图像内容（同步版本）"""
        raise NotImplementedError(f"analyze_vision() must be implemented by {self.provider_name} provider")
    
    @abstractmethod
    def format_prompt(self, text, vision_analysis, system):
        """格式化提示"""
        raise NotImplementedError(f"format_prompt() must be implemented by {self.provider_name} provider")
    
    def is_configured(self):
        """验证配置"""
        return bool(self.api_key and self.model)
    
    def get_name(self):
        """获取提供商名称"""
        return self.provider_name
    
    def build_vision_prompt(self, article_text, image_count):
        """构建视觉分析提示"""
        return f"""你正在分析一篇网络安全文章中的{image_count}张图片，以增强威胁情报分析。

文章上下文（前1000个字符）：
{article_text[:1000]}...

请分析这些图像并提供：
1. 截图中可见的技术细节（命令、文件路径、网络指标）
2. 显示的攻击技术或工具
3. 任何与MITRE ATT&CK相关的信息
4. 显示的系统配置或漏洞

专注于可操作的技术情报，以补充文章文本。"""