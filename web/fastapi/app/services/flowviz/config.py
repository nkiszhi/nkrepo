#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
FlowViz Configuration File - 使用数据库配置管理器
"""
import os
import sys
import configparser
import logging
try:
    from app.scripts.config_manager import config_manager
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("config_manager模块未找到,使用默认配置")
    config_manager = None

logger = logging.getLogger(__name__)

_flowviz_config = {}
_config_initialized = False

class FlowVizConfig:
    """FlowViz Configuration Class - 数据库配置版本"""
    
    # 默认值（如果数据库中没有则使用这些值）
    OPENAI_API_KEY = ''
    OPENAI_BASE_URL = 'https://api.openai.com/v1'
    OPENAI_MODEL = 'gpt-4o'
    CLAUDE_API_KEY = ''
    CLAUDE_BASE_URL = 'https://api.anthropic.com'
    CLAUDE_MODEL = 'claude-3-5-sonnet-20241022'
    DEFAULT_AI_PROVIDER = 'openai'
    
    # FlowViz 严格模式设置
    FLOW_STRICT_MODE = True
    REQUIRED_TECHNIQUE_IDS = True
    REQUIRE_MITRE_MAPPING = True
    VALIDATE_NODE_TYPES = True
    ENABLE_ATTACK_PHASE_GROUPING = True
    
    # 安全设置
    MAX_REQUEST_SIZE = 10485760
    MAX_IMAGE_SIZE = 3145728
    MAX_ARTICLE_SIZE = 5242880
    
    # 解析限制
    PARSE_MAX_NODES = 50
    PARSE_MAX_EDGES = 40
    PARSE_MIN_NODES = 10
    PARSE_MIN_EDGES = 8
    
    # OpenAI 增强设置
    OPENAI_MAX_TOKENS_MULTIPLIER = 2.5
    OPENAI_MIN_TOKENS = 8192
    OPENAI_MAX_TOKENS = 65536
    OPENAI_TEMPERATURE = 0.1
    
    # 默认节点和边类型
    DEFAULT_NODE_TYPES = ['action', 'tool', 'malware', 'asset', 'infrastructure', 'vulnerability', 'url', 'operator']
    DEFAULT_EDGE_TYPES = ['uses', 'targets', 'communicates_with', 'exploits', 'creates', 'modifies', 'leads_to', 'affects']
    
    @staticmethod
    def init_from_config_manager(config_manager):
        """从配置管理器初始化"""
        try:
            global _flowviz_config, _config_initialized
            
            # 从数据库获取FlowViz配置
            config_data = config_manager.get_flowviz_config()
            
            # 更新类属性
            if 'OPENAI_API_KEY' in config_data:
                FlowVizConfig.OPENAI_API_KEY = config_data['OPENAI_API_KEY']
            if 'OPENAI_BASE_URL' in config_data:
                FlowVizConfig.OPENAI_BASE_URL = config_data['OPENAI_BASE_URL']
            if 'OPENAI_MODEL' in config_data:
                FlowVizConfig.OPENAI_MODEL = config_data['OPENAI_MODEL']
            if 'CLAUDE_API_KEY' in config_data:
                FlowVizConfig.CLAUDE_API_KEY = config_data['CLAUDE_API_KEY']
            if 'CLAUDE_BASE_URL' in config_data:
                FlowVizConfig.CLAUDE_BASE_URL = config_data['CLAUDE_BASE_URL']
            if 'CLAUDE_MODEL' in config_data:
                FlowVizConfig.CLAUDE_MODEL = config_data['CLAUDE_MODEL']
            if 'DEFAULT_AI_PROVIDER' in config_data:
                FlowVizConfig.DEFAULT_AI_PROVIDER = config_data['DEFAULT_AI_PROVIDER']
            
            # 从数据库加载FlowViz特定设置
            try:
                # 尝试从数据库获取FlowViz设置
                flow_strict_mode = config_manager.get_config('flowviz', 'flow_strict_mode', 'true')
                FlowVizConfig.FLOW_STRICT_MODE = flow_strict_mode.lower() == 'true'
                
                required_technique_ids = config_manager.get_config('flowviz', 'required_technique_ids', 'true')
                FlowVizConfig.REQUIRED_TECHNIQUE_IDS = required_technique_ids.lower() == 'true'
                
                require_mitre_mapping = config_manager.get_config('flowviz', 'require_mitre_mapping', 'true')
                FlowVizConfig.REQUIRE_MITRE_MAPPING = require_mitre_mapping.lower() == 'true'
                
                validate_node_types = config_manager.get_config('flowviz', 'validate_node_types', 'true')
                FlowVizConfig.VALIDATE_NODE_TYPES = validate_node_types.lower() == 'true'
                
                enable_attack_phase_grouping = config_manager.get_config('flowviz', 'enable_attack_phase_grouping', 'true')
                FlowVizConfig.ENABLE_ATTACK_PHASE_GROUPING = enable_attack_phase_grouping.lower() == 'true'
                
            except Exception as e:
                logger.warning(f"从数据库读取FlowViz设置失败: {str(e)}，使用默认值")
            
            _config_initialized = True
            logger.info("✅ FlowViz配置已从数据库加载")
            logger.info(f"   - 严格模式: {FlowVizConfig.FLOW_STRICT_MODE}")
            logger.info(f"   - 需要技术ID: {FlowVizConfig.REQUIRED_TECHNIQUE_IDS}")
            logger.info(f"   - 默认提供商: {FlowVizConfig.DEFAULT_AI_PROVIDER}")
            
        except Exception as e:
            logger.error(f"从配置管理器加载FlowViz配置失败: {str(e)}")
            FlowVizConfig.load_from_config_ini()
    
    @staticmethod
    def load_from_config_ini():
        """从config.ini文件加载配置（备用方案）"""
        try:
            global _flowviz_config, _config_initialized
            cp = configparser.ConfigParser()
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
            
            if os.path.exists(config_path):
                cp.read(config_path, encoding='utf-8')
                
                if cp.has_section('flowviz'):
                    # 更新类属性
                    FlowVizConfig.OPENAI_API_KEY = cp.get('flowviz', 'openai_api_key', fallback='')
                    FlowVizConfig.OPENAI_BASE_URL = cp.get('flowviz', 'openai_base_url', fallback='https://api.openai.com/v1')
                    FlowVizConfig.OPENAI_MODEL = cp.get('flowviz', 'openai_model', fallback='gpt-4o')
                    FlowVizConfig.CLAUDE_API_KEY = cp.get('flowviz', 'claude_api_key', fallback='')
                    FlowVizConfig.CLAUDE_BASE_URL = cp.get('flowviz', 'claude_base_url', fallback='https://api.anthropic.com')
                    FlowVizConfig.CLAUDE_MODEL = cp.get('flowviz', 'claude_model', fallback='claude-3-5-sonnet-20241022')
                    FlowVizConfig.DEFAULT_AI_PROVIDER = cp.get('flowviz', 'default_ai_provider', fallback='openai')
                    
                    # FlowViz特定设置
                    FlowVizConfig.FLOW_STRICT_MODE = cp.get('flowviz', 'flow_strict_mode', fallback='true').lower() == 'true'
                    FlowVizConfig.REQUIRED_TECHNIQUE_IDS = cp.get('flowviz', 'required_technique_ids', fallback='true').lower() == 'true'
                    FlowVizConfig.REQUIRE_MITRE_MAPPING = cp.get('flowviz', 'require_mitre_mapping', fallback='true').lower() == 'true'
                    FlowVizConfig.VALIDATE_NODE_TYPES = cp.get('flowviz', 'validate_node_types', fallback='true').lower() == 'true'
                    FlowVizConfig.ENABLE_ATTACK_PHASE_GROUPING = cp.get('flowviz', 'enable_attack_phase_grouping', fallback='true').lower() == 'true'
                    
                    # 安全设置
                    FlowVizConfig.MAX_REQUEST_SIZE = int(cp.get('flowviz', 'max_request_size', fallback='10485760'))
                    FlowVizConfig.MAX_IMAGE_SIZE = int(cp.get('flowviz', 'max_image_size', fallback='3145728'))
                    FlowVizConfig.MAX_ARTICLE_SIZE = int(cp.get('flowviz', 'max_article_size', fallback='5242880'))
                    
                    # 解析限制
                    FlowVizConfig.PARSE_MAX_NODES = int(cp.get('flowviz', 'parse_max_nodes', fallback='50'))
                    FlowVizConfig.PARSE_MAX_EDGES = int(cp.get('flowviz', 'parse_max_edges', fallback='40'))
                    FlowVizConfig.PARSE_MIN_NODES = int(cp.get('flowviz', 'parse_min_nodes', fallback='10'))
                    FlowVizConfig.PARSE_MIN_EDGES = int(cp.get('flowviz', 'parse_min_edges', fallback='8'))
                    
                    # OpenAI增强设置
                    FlowVizConfig.OPENAI_MAX_TOKENS_MULTIPLIER = float(cp.get('flowviz', 'openai_max_tokens_multiplier', fallback='2.5'))
                    FlowVizConfig.OPENAI_MIN_TOKENS = int(cp.get('flowviz', 'openai_min_tokens', fallback='8192'))
                    FlowVizConfig.OPENAI_MAX_TOKENS = int(cp.get('flowviz', 'openai_max_tokens', fallback='65536'))
                    FlowVizConfig.OPENAI_TEMPERATURE = float(cp.get('flowviz', 'openai_temperature', fallback='0.1'))
                    
                    _config_initialized = True
                    logger.info("✅ FlowViz配置已从config.ini加载")
                else:
                    logger.warning("❌ config.ini中没有flowviz部分")
            else:
                logger.error(f"❌ 配置文件不存在: {config_path}")
        except Exception as e:
            logger.error(f"从config.ini加载FlowViz配置失败: {str(e)}")
    
    @staticmethod
    def get_config(key, default=None):
        """获取配置项"""
        try:
            if hasattr(FlowVizConfig, key):
                return getattr(FlowVizConfig, key)
            return default
        except Exception as e:
            logger.error(f"❌ 获取FlowViz配置{key}失败: {str(e)}")
            return default
    
    @staticmethod
    def get_provider_config(provider_id):
        """获取提供商配置"""
        try:
            if provider_id == 'openai':
                return {
                    'api_key': FlowVizConfig.OPENAI_API_KEY,
                    'base_url': FlowVizConfig.OPENAI_BASE_URL,
                    'model': FlowVizConfig.OPENAI_MODEL,
                    'provider_name': 'OpenAI',
                    'strict_mode': FlowVizConfig.FLOW_STRICT_MODE,
                    'required_technique_ids': FlowVizConfig.REQUIRED_TECHNIQUE_IDS,
                    'max_tokens': FlowVizConfig.OPENAI_MAX_TOKENS,
                    'temperature': FlowVizConfig.OPENAI_TEMPERATURE,
                    'max_tokens_multiplier': FlowVizConfig.OPENAI_MAX_TOKENS_MULTIPLIER,
                    'min_tokens': FlowVizConfig.OPENAI_MIN_TOKENS,
                }
            elif provider_id == 'claude':
                return {
                    'api_key': FlowVizConfig.CLAUDE_API_KEY,
                    'base_url': FlowVizConfig.CLAUDE_BASE_URL,
                    'model': FlowVizConfig.CLAUDE_MODEL,
                    'provider_name': 'Claude',
                    'strict_mode': FlowVizConfig.FLOW_STRICT_MODE,
                    'required_technique_ids': FlowVizConfig.REQUIRED_TECHNIQUE_IDS,
                    'max_tokens': 16000,
                    'temperature': 0.1,
                }
            else:
                return None
        except Exception as e:
            logger.error(f"❌ 获取提供商配置{provider_id}失败: {str(e)}")
            return None
    
    @staticmethod
    def get_default_provider():
        """获取默认提供商"""
        try:
            return FlowVizConfig.DEFAULT_AI_PROVIDER
        except:
            return 'openai'
    
    @staticmethod
    def get_strict_mode_settings():
        """获取严格模式设置"""
        return {
            'flow_strict_mode': FlowVizConfig.FLOW_STRICT_MODE,
            'required_technique_ids': FlowVizConfig.REQUIRED_TECHNIQUE_IDS,
            'require_mitre_mapping': FlowVizConfig.REQUIRE_MITRE_MAPPING,
            'validate_node_types': FlowVizConfig.VALIDATE_NODE_TYPES,
            'enable_attack_phase_grouping': FlowVizConfig.ENABLE_ATTACK_PHASE_GROUPING,
            'default_node_types': FlowVizConfig.DEFAULT_NODE_TYPES,
            'default_edge_types': FlowVizConfig.DEFAULT_EDGE_TYPES,
            'parse_limits': {
                'max_nodes': FlowVizConfig.PARSE_MAX_NODES,
                'max_edges': FlowVizConfig.PARSE_MAX_EDGES,
                'min_nodes': FlowVizConfig.PARSE_MIN_NODES,
                'min_edges': FlowVizConfig.PARSE_MIN_EDGES
            }
        }
    
    @staticmethod
    def get_all_config():
        """获取所有配置（用于调试）"""
        try:
            return {
                'openai': {
                    'api_key_set': bool(FlowVizConfig.OPENAI_API_KEY),
                    'model': FlowVizConfig.OPENAI_MODEL,
                    'base_url': FlowVizConfig.OPENAI_BASE_URL,
                    'max_tokens': FlowVizConfig.OPENAI_MAX_TOKENS,
                    'temperature': FlowVizConfig.OPENAI_TEMPERATURE,
                },
                'claude': {
                    'api_key_set': bool(FlowVizConfig.CLAUDE_API_KEY),
                    'model': FlowVizConfig.CLAUDE_MODEL,
                    'base_url': FlowVizConfig.CLAUDE_BASE_URL,
                },
                'default_provider': FlowVizConfig.DEFAULT_AI_PROVIDER,
                'strict_mode_settings': FlowVizConfig.get_strict_mode_settings(),
                'parsing': {
                    'min_nodes': FlowVizConfig.PARSE_MIN_NODES,
                    'max_nodes': FlowVizConfig.PARSE_MAX_NODES,
                    'min_edges': FlowVizConfig.PARSE_MIN_EDGES,
                    'max_edges': FlowVizConfig.PARSE_MAX_EDGES,
                },
                'security': {
                    'max_request_size': FlowVizConfig.MAX_REQUEST_SIZE,
                    'max_image_size': FlowVizConfig.MAX_IMAGE_SIZE,
                    'max_article_size': FlowVizConfig.MAX_ARTICLE_SIZE,
                },
                'openai_enhanced': {
                    'max_tokens_multiplier': FlowVizConfig.OPENAI_MAX_TOKENS_MULTIPLIER,
                    'min_tokens': FlowVizConfig.OPENAI_MIN_TOKENS,
                    'max_tokens': FlowVizConfig.OPENAI_MAX_TOKENS,
                    'temperature': FlowVizConfig.OPENAI_TEMPERATURE,
                }
            }
        except Exception as e:
            logger.error(f"❌ 获取所有配置失败: {str(e)}")
            return {}

# 初始化配置（通过配置管理器）
if config_manager:
    FlowVizConfig.init_from_config_manager(config_manager)
else:
    # 使用默认配置
    FlowVizConfig.init_from_dict({})