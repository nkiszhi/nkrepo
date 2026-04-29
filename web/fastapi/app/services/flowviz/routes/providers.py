from flask import Blueprint, jsonify
from ..providers.factory import ProviderFactory
import logging

bp = Blueprint('providers', __name__)

# 使用标准的Python logging
logger = logging.getLogger(__name__)

@bp.route('/providers', methods=['GET'])
def get_providers():
    """获取可用的AI提供商"""
    try:
        providers = ProviderFactory.get_available_providers()
        default_provider = ProviderFactory.get_default_provider()
        
        # 注意: 只记录提供商数量和名称，不记录包含api_key的配置信息
        # Python logging 的正确用法：第一个参数是消息字符串，后续参数是格式化参数
        logger.info('Providers request: availableCount=%d, defaultProvider=%s', 
                   len(providers), default_provider)
        
        return jsonify({
            'providers': providers,
            'defaultProvider': default_provider,
            'hasConfiguredProviders': len(providers) > 0
        })
        
    except Exception as error:
        # 错误日志的正确用法
        logger.error('Error getting providers: %s', str(error))
        # 注意: 不向用户暴露详细的错误信息
        return jsonify({
            'error': 'Failed to get available providers',
            'message': 'An error occurred while getting providers.'
        }), 500