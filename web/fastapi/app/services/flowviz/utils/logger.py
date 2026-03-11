import logging
import sys
from datetime import datetime

class Logger:
    """自定义日志记录器"""
    
    def __init__(self, name='flowviz'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # 文件处理器
        file_handler = logging.FileHandler('flowviz.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '[%(asctime)s] %(name)s %(levelname)s: %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def warn(self, message, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message, **kwargs):
        self.logger.error(message, extra=kwargs)

# 创建全局日志记录器
logger = Logger()