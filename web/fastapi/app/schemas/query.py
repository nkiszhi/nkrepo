"""
查询相关的Pydantic模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class QueryRequest(BaseModel):
    """查询请求"""
    keyword: Optional[str] = Field(None, description="关键词")
    category: Optional[str] = Field(None, description="分类")
    family: Optional[str] = Field(None, description="家族")
    platform: Optional[str] = Field(None, description="平台")


class QueryResponse(BaseModel):
    """查询响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="查询结果")
    total: Optional[int] = Field(None, description="总数")


class SearchRequest(BaseModel):
    """搜索请求"""
    sha256: Optional[str] = Field(None, description="SHA256")
    md5: Optional[str] = Field(None, description="MD5")
    filename: Optional[str] = Field(None, description="文件名")
