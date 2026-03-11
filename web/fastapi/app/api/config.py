"""
配置管理API
"""
from fastapi import APIRouter, HTTPException, Depends
from app.api.auth import get_current_user
from app.core import settings
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
import configparser

router = APIRouter()
logger = logging.getLogger(__name__)


class ConfigResponse(BaseModel):
    """配置响应"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    section: str
    key: str
    value: str


@router.get("/config/all", response_model=ConfigResponse)
async def get_all_config(current_user: dict = Depends(get_current_user)):
    """
    获取所有配置
    """
    try:
        config_file = "config.ini"
        if not os.path.exists(config_file):
            return ConfigResponse(
                success=False,
                message="配置文件不存在"
            )
        
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        
        config_dict = {}
        for section in config.sections():
            config_dict[section] = dict(config.items(section))
        
        return ConfigResponse(
            success=True,
            message="获取成功",
            data=config_dict
        )
        
    except Exception as e:
        logger.error(f"获取配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/update", response_model=ConfigResponse)
async def update_config(
    request: ConfigUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    更新配置
    """
    try:
        config_file = "config.ini"
        
        config = configparser.ConfigParser()
        if os.path.exists(config_file):
            config.read(config_file, encoding='utf-8')
        
        # 更新配置
        if not config.has_section(request.section):
            config.add_section(request.section)
        
        config.set(request.section, request.key, request.value)
        
        # 保存配置
        with open(config_file, 'w', encoding='utf-8') as f:
            config.write(f)
        
        logger.info(f"配置已更新: [{request.section}] {request.key} = {request.value}")
        
        return ConfigResponse(
            success=True,
            message="更新成功"
        )
        
    except Exception as e:
        logger.error(f"更新配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/env")
async def get_env_config():
    """
    获取环境变量配置
    """
    return {
        "success": True,
        "data": {
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
            "debug": settings.DEBUG,
            "host": settings.HOST,
            "port": settings.PORT,
            "upload_dir": settings.UPLOAD_DIR,
            "log_level": settings.LOG_LEVEL
        }
    }
