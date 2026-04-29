# flowviz/providers/factory.py
from .openai import OpenAIProvider
from .claude import ClaudeProvider
from ..config import FlowVizConfig
import logging

logger = logging.getLogger(__name__)

class ProviderFactory:
    """AI Provider Factory - 支持OpenAI和Claude，增强严格模式"""
    
    @staticmethod
    def create(provider_name='openai', config=None):
        """创建提供商实例"""
        if config is None:
            config = ProviderFactory.get_provider_config(provider_name)
        
        normalized = provider_name.lower().strip()
        
        if normalized in ['openai', 'gpt']:
            return OpenAIProvider(config)
        elif normalized in ['claude', 'anthropic']:
            return ClaudeProvider(config)
        else:
            raise ValueError(f"不支持的提供商: {provider_name}")
    
    @staticmethod
    def get_available_providers():
        """获取可用的提供商列表"""
        providers = []
        
        # 检查OpenAI配置
        openai_config = FlowVizConfig.get_provider_config('openai')
        if openai_config and openai_config.get('api_key'):
            providers.append({
                'id': 'openai',
                'name': 'OpenAI',
                'displayName': 'OpenAI',
                'models': OpenAIProvider.get_supported_models(),
                'defaultModel': openai_config.get('model', 'gpt-4o'),
                'configured': True,
                'description': 'OpenAI GPT系列模型，支持多种任务',
                'supports_strict_mode': True
            })
        
        # 检查Claude配置
        claude_config = FlowVizConfig.get_provider_config('claude')
        if claude_config and claude_config.get('api_key'):
            providers.append({
                'id': 'anthropic',
                'name': 'Anthropic',
                'displayName': 'Anthropic',
                'models': ClaudeProvider.get_supported_models(),
                'defaultModel': claude_config.get('model', 'claude-3-5-sonnet-20241022'),
                'configured': True,
                'description': 'Anthropic Claude模型，具有高级推理能力',
                'supports_strict_mode': True
            })
        
        # 注意: 只记录提供商ID列表，不记录包含api_key的配置信息
        logger.info(f"可用提供商: {[p['id'] for p in providers]}")
        return providers
    
    @staticmethod
    def get_default_provider():
        """基于配置获取默认提供商"""
        # 从配置检查显式默认值
        default_provider = FlowVizConfig.get_config('DEFAULT_AI_PROVIDER', 'openai').lower().strip()
        
        # 验证提供商是否已配置
        available_providers = ProviderFactory.get_available_providers()
        available_ids = [p['id'] for p in available_providers]
        
        if default_provider in ['openai', 'gpt'] and 'openai' in available_ids:
            return 'openai'
        elif default_provider in ['anthropic', 'claude'] and 'anthropic' in available_ids:
            return 'anthropic'
        elif available_providers:
            # 返回第一个可用的提供商
            return available_providers[0]['id']
        
        return None
    
    @staticmethod
    def get_provider_config(provider_id):
        """从FlowVizConfig获取提供商配置"""
        normalized = provider_id.lower().strip()
        
        if normalized in ['openai', 'gpt']:
            return FlowVizConfig.get_provider_config('openai')
        elif normalized in ['anthropic', 'claude']:
            return FlowVizConfig.get_provider_config('claude')
        else:
            raise ValueError(f"未知提供商: {provider_id}")
    
    @staticmethod
    def has_configured_providers():
        """检查是否至少有一个提供商已配置"""
        return len(ProviderFactory.get_available_providers()) > 0
    
    @staticmethod
    def get_provider_info(provider_id):
        """按ID获取提供商信息"""
        providers = ProviderFactory.get_available_providers()
        for provider in providers:
            if provider['id'] == provider_id:
                return provider
        return None
    
    @staticmethod
    def get_strict_mode_prompt(strict_mode=True):
        """获取严格模式提示词部分"""
        if strict_mode:
            return """STRICT REQUIREMENTS:
1. ALL action nodes MUST have valid MITRE ATT&CK technique IDs (format: T####)
2. Nodes MUST be organized by attack phase: Initial Access, Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, Exfiltration, Command and Control
3. EACH node must include: technique_id, tactic, platform, and data_sources where applicable
4. Use ONLY valid MITRE ATT&CK techniques from the latest framework
5. Include operator nodes (AND/OR) for complex attack logic
6. Ensure chronological flow with clear progression
7. Validate all technical indicators against known patterns"""
        else:
            return """GUIDELINES:
1. Try to include MITRE ATT&CK technique IDs where possible
2. Organize nodes logically by attack progression
3. Include relevant technical details
4. Create a clear, understandable attack flow"""