"""
API路由初始化
"""
from fastapi import APIRouter
from app.api import auth, detect, attck, flowviz, query, config, flowviz_streaming, av_scan

# 创建路由实例
auth_router = auth.router
detect_router = detect.router
attck_router = attck.router
flowviz_router = flowviz.router
query_router = query.router
config_router = config.router
flowviz_streaming_router = flowviz_streaming.router
av_scan_router = av_scan.router

__all__ = [
    'auth_router',
    'detect_router',
    'attck_router',
    'flowviz_router',
    'query_router',
    'config_router',
    'flowviz_streaming_router',
    'av_scan_router'
]
