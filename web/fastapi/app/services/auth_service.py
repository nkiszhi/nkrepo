"""
认证服务 - 完全按照旧Flask方式实现
"""
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from app.scripts.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# JWT配置
JWT_SECRET = 'your-secret-key-change-in-production'
JWT_EXPIRE_HOURS = 24
JWT_ALGORITHM = 'HS256'

# 使用旧的ConfigManager
config_manager = ConfigManager()


class AuthService:
    """认证服务类 - 完全按照旧Flask实现"""
    
    def __init__(self):
        self.config_manager = config_manager
    
    def authenticate_user(self, username: str, password: str) -> Dict:
        """
        认证用户 - 完全按照旧Flask的user_login函数实现
        返回: {'success': bool, 'message': str, 'token': str, 'username': str}
        """
        try:
            if not all([username, password]):
                return {
                    'success': False,
                    'message': '用户名和密码不能为空'
                }
            
            # 从数据库获取存储的用户名和密码
            stored_username = self.config_manager.get_config('fixed_user', 'username', 'admin')
            stored_password_hash = self.config_manager.get_config('fixed_user', 'password', '')
            
            # 验证用户名
            if username != stored_username:
                return {
                    'success': False,
                    'message': '用户名或密码错误'
                }
            
            # 验证密码
            if not stored_password_hash:
                # 如果数据库中没有密码,使用config.ini中的密码(向后兼容)
                import configparser
                cp = configparser.ConfigParser()
                cp.read('config.ini', encoding='utf-8')
                fallback_password = cp.get('fixed_user', 'password', fallback='123456')
                if password != fallback_password:
                    return {
                        'success': False,
                        'message': '用户名或密码错误'
                    }
            else:
                # 使用bcrypt验证密码
                if not self.config_manager.verify_password(password, stored_password_hash):
                    return {
                        'success': False,
                        'message': '用户名或密码错误'
                    }
            
            # 生成JWT token
            expire_time = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
            token = jwt.encode(
                {
                    'user_id': 1,
                    'username': username,
                    'exp': expire_time
                },
                JWT_SECRET,
                algorithm=JWT_ALGORITHM
            )
            
            return {
                'success': True,
                'token': token,
                'username': username,
                'message': '登录成功'
            }
            
        except Exception as e:
            logger.error(f"登录接口异常: {str(e)}")
            return {
                'success': False,
                'message': f'登录失败: {str(e)}'
            }
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """验证JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token已过期")
            return None
        except jwt.InvalidTokenError:
            logger.warning("无效的Token")
            return None
