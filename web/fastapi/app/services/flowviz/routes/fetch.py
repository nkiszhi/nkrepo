from flask import Blueprint, request, jsonify
import requests
import base64
from urllib.parse import urlparse
import time
import mimetypes
import imghdr
from bs4 import BeautifulSoup
import logging

bp = Blueprint('fetch', __name__)
logger = logging.getLogger(__name__)

# 添加简单的readability实现，避免依赖问题
class SimpleReadability:
    """简单的Readability实现，避免外部依赖"""
    
    @staticmethod
    def extract(html, url=None):
        """提取文章主要内容"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 获取标题
            title = soup.title.string if soup.title else ''
            
            # 尝试找到主要内容
            # 1. 查找article标签
            article = soup.find('article')
            if article:
                return {
                    'title': title,
                    'content': str(article),
                    'summary': article.get_text()[:200]
                }
            
            # 2. 查找主要内容容器
            main_content = soup.find(['main', 'div#content', 'div.content', 'div.post-content'])
            if main_content:
                return {
                    'title': title,
                    'content': str(main_content),
                    'summary': main_content.get_text()[:200]
                }
            
            # 3. 回退到body
            body = soup.body
            if body:
                return {
                    'title': title,
                    'content': str(body),
                    'summary': body.get_text()[:200]
                }
            
            return {
                'title': title,
                'content': html,
                'summary': html[:200]
            }
            
        except Exception as e:
            logger.error(f'Readability error: {e}')
            return {
                'title': '',
                'content': html,
                'summary': html[:200]
            }

def validate_url(url_str):
    """验证URL"""
    try:
        parsed = urlparse(url_str)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError('Invalid URL')
        if parsed.scheme not in ['http', 'https']:
            raise ValueError('Only HTTP and HTTPS URLs are allowed')
        return url_str
    except Exception as e:
        raise ValueError(f'Invalid URL: {str(e)}')

def secure_fetch(url, options=None):
    """安全获取网页内容"""
    if options is None:
        options = {}
    
    timeout = options.get('timeout', 30)
    headers = options.get('headers', {})
    headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        return response
    except requests.RequestException as e:
        logger.error(f'Fetch error: {e}')
        raise

@bp.route('/fetch-image', methods=['GET'])
def fetch_image():
    """安全图片获取端点"""
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400
    
    logger.debug(f'Fetching image from: {url}')
    
    try:
        # 验证URL
        validated_url = validate_url(url)
        
        # 安全获取图片
        response = secure_fetch(validated_url, {
            'timeout': 15,
            'headers': {'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'}
        })
        
        logger.debug(f'Image response status: {response.status_code}')
        
        if not response.ok:
            logger.error(f'Failed to fetch image: {response.status_code}')
            return jsonify({'error': f'Failed to fetch image: {response.reason}'}), response.status_code
        
        # 获取图片数据
        buffer = response.content
        
        if len(buffer) > 3 * 1024 * 1024:  # 3MB限制
            return jsonify({'error': 'Image too large. Maximum size is 3MB.'}), 413
        
        # 检测内容类型
        content_type = response.headers.get('Content-Type', '')
        
        # 如果没有Content-Type，尝试检测
        if not content_type or not content_type.startswith('image/'):
            image_type = imghdr.what(None, h=buffer)
            if image_type:
                content_type = f'image/{image_type}'
            else:
                # 尝试通过文件扩展名判断
                parsed_url = urlparse(url)
                ext = parsed_url.path.split('.')[-1].lower() if '.' in parsed_url.path else ''
                ext_to_mime = {
                    'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg',
                    'png': 'image/png',
                    'gif': 'image/gif',
                    'webp': 'image/webp',
                    'bmp': 'image/bmp'
                }
                content_type = ext_to_mime.get(ext, 'image/jpeg')
        
        # 验证是否是图片类型
        allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp']
        if not any(content_type.startswith(t) for t in allowed_types):
            return jsonify({'error': f'Unsupported image format: {content_type}'}), 400
        
        # 转换为base64
        base64_data = base64.b64encode(buffer).decode('utf-8')
        
        logger.info(f'Image fetched: {content_type}, {len(base64_data)//1024}KB')
        return jsonify({'base64': base64_data, 'mediaType': content_type})
        
    except Exception as error:
        logger.error(f'Image fetch error: {error}')
        return jsonify({
            'error': 'Failed to fetch image',
            'details': str(error)
        }), 500

@bp.route('/fetch-article', methods=['GET'])
def fetch_article():
    """安全文章获取端点"""
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400
    
    logger.debug(f'Fetching article from: {url}')
    
    try:
        # 验证URL
        validated_url = validate_url(url)
        
        # 安全获取文章
        response = secure_fetch(validated_url, {
            'timeout': 30,
            'headers': {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
        })
        
        logger.debug(f'Article response status: {response.status_code}')
        
        if not response.ok:
            logger.error(f'Failed to fetch article: {response.status_code}')
            return jsonify({'error': f'Failed to fetch article: {response.reason}'}), response.status_code
        
        # 验证内容类型
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type.lower():
            return jsonify({'error': 'Invalid content type. Only HTML content is supported.'}), 400
        
        # 获取HTML
        html = response.text
        
        if len(html) > 5 * 1024 * 1024:  # 5MB限制
            return jsonify({'error': 'Content too large. Maximum size is 5MB.'}), 413
        
        # 使用简单的readability解析
        result = SimpleReadability.extract(html, validated_url)
        
        if not result.get('content'):
            logger.warn('Readability parsing failed, falling back to raw HTML')
            return jsonify({'contents': html})
        
        # 构建增强的HTML
        title = result.get('title', 'Article')
        enhanced_html = f"""
        <html>
            <head><title>{title}</title></head>
            <body>
                <h1>{title}</h1>
                <div class="content">{result['content']}</div>
            </body>
        </html>
        """
        
        # 提取元数据
        soup = BeautifulSoup(html, 'html.parser')
        meta_author = soup.find('meta', {'name': 'author'})
        byline = meta_author.get('content') if meta_author else None
        
        if not byline:
            author_tag = soup.find(['span.author', 'div.author', 'p.author'])
            byline = author_tag.get_text() if author_tag else None
        
        logger.info(f'Successfully parsed article: "{title}", content length: {len(enhanced_html)} characters')
        
        return jsonify({
            'contents': enhanced_html,
            'metadata': {
                'title': title,
                'byline': byline,
                'excerpt': result.get('summary', '')[:200],
                'length': len(enhanced_html),
                'readTime': max(1, len(enhanced_html) // 2000)
            }
        })
        
    except Exception as error:
        logger.error(f'Article fetch error: {error}')
        return jsonify({
            'error': 'Failed to fetch article',
            'details': str(error)
        }), 500