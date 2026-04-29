from flask import Blueprint, request, jsonify
from ..providers.factory import ProviderFactory
import re
import logging

bp = Blueprint('vision', __name__)
logger = logging.getLogger(__name__)

@bp.route('/vision-analysis', methods=['POST'])
def vision_analysis():
    """Vision analysis endpoint - English version"""
    try:
        data = request.get_json()
        images = data.get('images')
        article_text = data.get('articleText')
        
        if not images or not isinstance(images, list) or len(images) == 0:
            return jsonify({'error': 'Missing or invalid images array'}), 400
        
        if not article_text or not isinstance(article_text, str):
            return jsonify({'error': 'Missing or invalid articleText'}), 400
        
        logger.info(f'Vision analysis request: {len(images)} images, {len(article_text)} chars of text')
        
        default_provider = ProviderFactory.get_default_provider()
        if not default_provider:
            return jsonify({'error': 'No AI providers configured'}), 500
        
        provider_config = ProviderFactory.get_provider_config(default_provider)
        ai_provider = ProviderFactory.create(default_provider, provider_config)
        
        # 使用统一的视觉分析提示词
        result = ai_provider.analyze_vision(images, article_text)
        
        confidence = assess_confidence_english(result['analysisText'], len(images))
        
        logger.info(f'Vision analysis completed: {len(result["analysisText"])} chars, {confidence} confidence')
        
        return jsonify({
            'analysisText': result['analysisText'],
            'confidence': confidence,
            'relevantImages': images,
            'tokensUsed': result.get('tokensUsed', 0)
        })
        
    except Exception as error:
        logger.error('Vision analysis error:', str(error))
        # 注意: 不向用户暴露详细的错误信息，只返回通用错误消息
        return jsonify({
            'error': 'Vision analysis failed',
            'message': 'An error occurred during vision analysis. Please try again later.'
        }), 500

def assess_confidence_english(analysis_text, image_count):
    """Assess confidence level - English version"""
    if not analysis_text or len(analysis_text) < 100:
        return 'low'
    
    technical_indicators = [
        re.compile(r'T\d{4}'),  # MITRE technique IDs
        re.compile(r'CVE-\d{4}-\d+'),  # CVE references
        re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),  # IP addresses
        re.compile(r'[a-f0-9]{32,}')  # Hashes
    ]
    
    matches = sum(len(pattern.findall(analysis_text)) for pattern in technical_indicators)
    
    if matches >= 3 and image_count >= 2:
        return 'high'
    if matches >= 1 or image_count >= 1:
        return 'medium'
    return 'low'