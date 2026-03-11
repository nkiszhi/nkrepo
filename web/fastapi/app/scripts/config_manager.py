#!/usr/bin/env python3
# -*-coding: utf-8-*-

import os
import configparser
import pymysql
import bcrypt
from datetime import datetime
import logging

logger = logging.getLogger('config_manager')

class ConfigManager:
    """配置管理器：从数据库读取配置"""
    
    def __init__(self):
        # 读取config.ini获取数据库连接信息
        # config.ini在项目根目录,需要向上两级
        self.config_file = os.path.join(os.path.dirname(__file__), '../../config.ini')
        self.cp = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            self.cp.read(self.config_file, encoding='utf-8')
        
        # 数据库连接配置
        self.db_config = {
            'host': self.cp.get('mysql', 'host', fallback='localhost'),
            'user': self.cp.get('mysql', 'user', fallback='root'),
            'password': self.cp.get('mysql', 'passwd', fallback=''),
            'database': self.cp.get('mysql', 'db_web', fallback='webdatadb'),
            'charset': self.cp.get('mysql', 'charset', fallback='utf8mb4')
        }
        
        # 配置缓存
        self.config_cache = {}
        logger.info(f"ConfigManager初始化完成，连接数据库: {self.db_config['host']}/{self.db_config['database']}")
    
    def get_connection(self):
        """获取数据库连接"""
        try:
            return pymysql.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                charset=self.db_config['charset'],
                cursorclass=pymysql.cursors.DictCursor
            )
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            return None
    
    def load_all_configs(self, force_refresh=False):
        """从数据库加载所有配置到缓存"""
        try:
            conn = self.get_connection()
            if not conn:
                logger.warning("无法连接数据库，使用config.ini配置")
                return self.load_from_config_ini()
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT config_key, config_value, config_module FROM system_config")
                results = cursor.fetchall()
                
                self.config_cache = {}
                for row in results:
                    module = row['config_module']
                    key = row['config_key']
                    value = row['config_value']
                    
                    if module not in self.config_cache:
                        self.config_cache[module] = {}
                    self.config_cache[module][key] = value
                
                logger.info(f"从数据库加载了{len(results)}个配置项")
                return self.config_cache
                
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}，使用config.ini")
            return self.load_from_config_ini()
        finally:
            if 'conn' in locals() and conn:
                conn.close()
    
    def load_from_config_ini(self):
        """从config.ini文件加载配置（备用方案）"""
        configs = {}
        
        # 加载fixed_user配置
        if self.cp.has_section('fixed_user'):
            configs['fixed_user'] = {
                'username': self.cp.get('fixed_user', 'username', fallback='admin'),
                'password': self.cp.get('fixed_user', 'password', fallback='123456')
            }
        
        # 加载API配置
        if self.cp.has_section('API'):
            configs['api'] = {
                'vt_key': self.cp.get('API', 'vt_key', fallback='')
            }
        
        # 加载flowviz配置
        if self.cp.has_section('flowviz'):
            configs['ai_provider'] = {
                'openai_api_key': self.cp.get('flowviz', 'openai_api_key', fallback=''),
                'openai_base_url': self.cp.get('flowviz', 'openai_base_url', fallback='https://api.closeai-asia.com/v1'),
                'openai_model': self.cp.get('flowviz', 'openai_model', fallback='gpt-4o'),
                'claude_api_key': self.cp.get('flowviz', 'claude_api_key', fallback=''),
                'claude_base_url': self.cp.get('flowviz', 'claude_base_url', fallback='https://api.openai-proxy.org/anthropic'),
                'claude_model': self.cp.get('flowviz', 'claude_model', fallback='claude-3-5-sonnet-20240620'),
                'default_ai_provider': self.cp.get('flowviz', 'default_ai_provider', fallback='openai')
            }
        
        self.config_cache = configs
        return configs
    
    def get_config(self, module, key, default=None):
        """获取单个配置项"""
        if not self.config_cache:
            self.load_all_configs()
        
        if module in self.config_cache and key in self.config_cache[module]:
            return self.config_cache[module][key]
        
        # 如果数据库中不存在，尝试从config.ini获取
        return self.get_from_config_ini(module, key, default)
    
    def get_from_config_ini(self, module, key, default=None):
        """从config.ini获取配置"""
        if module == 'fixed_user' and self.cp.has_section('fixed_user'):
            return self.cp.get('fixed_user', key, fallback=default)
        elif module == 'api' and self.cp.has_section('API'):
            if key == 'vt_key':
                return self.cp.get('API', 'vt_key', fallback=default)
        elif module == 'ai_provider' and self.cp.has_section('flowviz'):
            if key in ['openai_api_key', 'openai_base_url', 'openai_model', 
                      'claude_api_key', 'claude_base_url', 'claude_model', 'default_ai_provider']:
                return self.cp.get('flowviz', key, fallback=default)
        return default
    
    def update_config(self, module, key, value, description=None):
        """更新配置到数据库"""
        try:
            conn = self.get_connection()
            if not conn:
                return False, "数据库连接失败"
            
            with conn.cursor() as cursor:
                # 检查是否存在
                cursor.execute(
                    "SELECT id FROM system_config WHERE config_key = %s AND config_module = %s",
                    (key, module)
                )
                exists = cursor.fetchone()
                
                if exists:
                    # 更新现有配置
                    sql = """
                    UPDATE system_config 
                    SET config_value = %s, description = COALESCE(%s, description), updated_at = NOW()
                    WHERE config_key = %s AND config_module = %s
                    """
                    cursor.execute(sql, (value, description, key, module))
                else:
                    # 插入新配置
                    sql = """
                    INSERT INTO system_config (config_key, config_value, config_module, description)
                    VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (key, value, module, description))
                
                conn.commit()
                
                # 更新缓存
                if module not in self.config_cache:
                    self.config_cache[module] = {}
                self.config_cache[module][key] = value
                
                logger.info(f"配置更新成功: {module}.{key}")
                return True, "更新成功"
                
        except Exception as e:
            logger.error(f"更新配置失败: {str(e)}")
            return False, f"更新失败: {str(e)}"
        finally:
            if 'conn' in locals() and conn:
                conn.close()
    
    def update_password(self, username, new_password):
        """更新密码（自动加密）"""
        try:
            # 使用bcrypt加密密码
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            return self.update_config('fixed_user', 'password', hashed_password.decode('utf-8'), '系统管理员密码')
        except Exception as e:
            logger.error(f"密码更新失败: {str(e)}")
            return False, f"密码更新失败: {str(e)}"
    
    def verify_password(self, password, hashed_password):
        """验证密码"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"密码验证失败: {str(e)}")
            return False
    
    def get_flowviz_config(self):
        """专门为flowviz模块获取配置"""
        configs = self.get_config('ai_provider', 'dummy', {})
        if not configs:
            configs = {}
        
        return {
            'OPENAI_API_KEY': self.get_config('ai_provider', 'openai_api_key', ''),
            'OPENAI_BASE_URL': self.get_config('ai_provider', 'openai_base_url', 'https://api.closeai-asia.com/v1'),
            'OPENAI_MODEL': self.get_config('ai_provider', 'openai_model', 'gpt-4o'),
            'CLAUDE_API_KEY': self.get_config('ai_provider', 'claude_api_key', ''),
            'CLAUDE_BASE_URL': self.get_config('ai_provider', 'claude_base_url', 'https://api.openai-proxy.org/anthropic'),
            'CLAUDE_MODEL': self.get_config('ai_provider', 'claude_model', 'claude-3-5-sonnet-20240620'),
            'DEFAULT_AI_PROVIDER': self.get_config('ai_provider', 'default_ai_provider', 'openai')
        }
    
    def get_all_configs_formatted(self):
        """获取所有配置（格式化）"""
        self.load_all_configs()
        
        return {
            'fixed_user': {
                'username': self.get_config('fixed_user', 'username', 'admin'),
                'password': ''  # 密码不返回明文
            },
            'api': {
                'vt_key': self.get_config('api', 'vt_key', '')
            },
            'ai_provider': {
                'openai_api_key': self.get_config('ai_provider', 'openai_api_key', ''),
                'openai_base_url': self.get_config('ai_provider', 'openai_base_url', 'https://api.closeai-asia.com/v1'),
                'openai_model': self.get_config('ai_provider', 'openai_model', 'gpt-4o'),
                'claude_api_key': self.get_config('ai_provider', 'claude_api_key', ''),
                'claude_base_url': self.get_config('ai_provider', 'claude_base_url', 'https://api.openai-proxy.org/anthropic'),
                'claude_model': self.get_config('ai_provider', 'claude_model', 'claude-3-5-sonnet-20240620'),
                'default_ai_provider': self.get_config('ai_provider', 'default_ai_provider', 'openai')
            },
            '_source': 'database',
            '_last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# 创建全局配置管理器实例
config_manager = ConfigManager()