# flowviz/utils/security.py
import ipaddress
import socket
from urllib.parse import urljoin, urlparse
from functools import wraps
from flask import request, jsonify
from datetime import datetime, timedelta
import requests
import time
import random
import logging

logger = logging.getLogger(__name__)

ALLOWED_URL_SCHEMES = {'http', 'https'}
BLOCKED_HOSTNAMES = {
    'localhost',
    'localhost.localdomain',
    'ip6-localhost',
    'ip6-loopback',
}
MAX_SAFE_REDIRECTS = 5


def _is_public_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


def _validate_public_host(hostname: str):
    if not hostname:
        raise ValueError('URL hostname is required')

    normalized = hostname.rstrip('.').lower()
    if normalized in BLOCKED_HOSTNAMES or normalized.endswith('.localhost'):
        raise ValueError('Internal hostnames are not allowed')

    try:
        ip_obj = ipaddress.ip_address(normalized)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        if not ip_obj.is_global:
            raise ValueError('Only public IP addresses are allowed')
        return

    try:
        address_infos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f'Unable to resolve hostname: {hostname}') from exc

    resolved_ips = {info[4][0] for info in address_infos}
    if not resolved_ips:
        raise ValueError(f'Unable to resolve hostname: {hostname}')
    if any(not _is_public_ip(ip) for ip in resolved_ips):
        raise ValueError('Host resolves to a non-public address')

def validate_url(url_str):
    """Validate URL to prevent SSRF attacks"""
    try:
        parsed = urlparse(url_str)

        if not parsed.scheme or not parsed.netloc:
            raise ValueError('Invalid URL')

        if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
            raise ValueError('Only HTTP and HTTPS URLs are allowed')

        _validate_public_host(parsed.hostname)
        return url_str
    except Exception as e:
        raise ValueError(f'Invalid URL: {str(e)}')


def _request_with_safe_redirects(url, headers, timeout, stream=True, verify=True, max_redirects=MAX_SAFE_REDIRECTS):
    current_url = validate_url(url)

    for _ in range(max_redirects + 1):
        response = requests.get(
            current_url,
            headers=headers,
            timeout=timeout,
            stream=stream,
            verify=verify,
            allow_redirects=False,
        )

        if not response.is_redirect:
            return response

        location = response.headers.get('Location')
        if not location:
            return response

        current_url = validate_url(urljoin(current_url, location))

    raise ValueError('Too many redirects')

def secure_fetch(url, options=None):
    """Secure web content fetching (optimized version)"""
    if options is None:
        options = {}
    
    timeout = options.get('timeout', 30)
    max_size = options.get('max_size', 10 * 1024 * 1024)  # 10MB default
    headers = options.get('headers', {})
    
    # Add common browser headers to avoid blocking
    headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    })
    
    try:
        # Random delay to avoid being identified as a crawler
        time.sleep(random.uniform(0.5, 1.5))
        
        response = _request_with_safe_redirects(
            url,
            headers=headers,
            timeout=timeout,
            stream=True,
            verify=True,
        )
        
        # Check response size
        if 'Content-Length' in response.headers:
            content_length = int(response.headers['Content-Length'])
            if content_length > max_size:
                raise ValueError(f'Response too large: {content_length} > {max_size}')
        
        return response
    except requests.exceptions.Timeout:
        raise ValueError(f'Request timeout after {timeout} seconds')
    except requests.exceptions.SSLError:
        # If SSL error, try without certificate verification
        try:
            response = _request_with_safe_redirects(
                url,
                headers=headers,
                timeout=timeout,
                stream=True,
                verify=False,
            )
            return response
        except:
            raise ValueError('SSL certificate verification failed')
    except requests.exceptions.RequestException as e:
        raise ValueError(f'Request failed: {str(e)}')

class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key, limit, period):
        """Check if request is allowed"""
        now = datetime.now()
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean expired records
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < timedelta(seconds=period)
        ]
        
        if len(self.requests[key]) < limit:
            self.requests[key].append(now)
            return True
        
        return False

rate_limiter = RateLimiter()

def rate_limit(limit_type):
    """Rate limiting decorator"""
    limits = {
        'streaming': (100, 86400),  # 100 times/day
        'images': (500, 86400),     # 500 times/day
        'articles': (200, 86400)    # 200 times/day
    }
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = request.remote_addr or 'unknown'
            key = f"{limit_type}:{client_ip}"
            
            limit, period = limits.get(limit_type, (100, 86400))
            
            if not rate_limiter.is_allowed(key, limit, period):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Limit: {limit} per day'
                }), 429
            
            return f(*args, **kwargs)
        return wrapped
    return decorator

def handle_fetch_error(error, context='resource'):
    """Common error handler for fetch operations"""
    logger.error(f"Error fetching {context}: {error}")
    
    error_msg = str(error)
    
    if 'Request timeout' in error_msg:
        return {'status': 408, 'error': 'Request timeout'}
    elif 'Invalid URL' in error_msg:
        return {'status': 400, 'error': error_msg}
    elif 'ENOTFOUND' in error_msg or 'Name or service not known' in error_msg:
        return {'status': 404, 'error': f'{context} not found'}
    elif 'Connection refused' in error_msg:
        return {'status': 503, 'error': 'Service unavailable'}
    elif 'SSL certificate' in error_msg:
        return {'status': 526, 'error': 'Invalid SSL certificate'}
    else:
        return {
            'status': 500,
            'error': f'Failed to fetch {context}',
            'details': error_msg
        }
