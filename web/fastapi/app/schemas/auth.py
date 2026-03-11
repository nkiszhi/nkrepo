"""
认证相关的Pydantic模型
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., min_length=1, max_length=50, description="用户名")
    password: str = Field(..., min_length=1, max_length=100, description="密码")


class LoginResponse(BaseModel):
    """登录响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    token: Optional[str] = Field(None, description="JWT令牌")
    username: Optional[str] = Field(None, description="用户名")
    user_id: Optional[int] = Field(None, description="用户ID")


class TokenPayload(BaseModel):
    """Token载荷"""
    user_id: int
    username: str
    exp: datetime
