#!/usr/bin/env python3
# -*-coding: utf-8-*-
"""
SSE (Server-Sent Events) 工具函数
"""
import json

def sse_message(event_type: str, data: dict) -> str:
    """生成SSE格式消息"""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"

def sse_done() -> str:
    """生成SSE完成消息"""
    return "data: [DONE]\n\n"

def sse_progress(stage: str, message: str, percentage: int = 0) -> str:
    """生成进度消息"""
    return sse_message('progress', {
        'stage': stage,
        'message': message,
        'percentage': percentage
    })