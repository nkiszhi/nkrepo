"""
检测相关的Pydantic模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class UploadResponse(BaseModel):
    """文件上传响应"""
    original_filename: str = Field(..., description="原始文件名")
    query_result: Dict[str, Any] = Field(..., description="查询结果")
    file_size: str = Field(..., description="文件大小")
    exe_result: Dict[str, Any] = Field(..., description="检测结果")
    VT_API: Optional[str] = Field(None, description="VirusTotal API")
    sha256: Optional[str] = Field(None, description="SHA256哈希")


class DetectionResponse(BaseModel):
    """检测响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    data: Optional[Dict[str, Any]] = Field(None, description="检测数据")


class DomainDetectRequest(BaseModel):
    """域名检测请求"""
    url: str = Field(..., description="域名或URL")


class DomainDetectResponse(BaseModel):
    """域名检测响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    result: Optional[Dict[str, Any]] = Field(None, description="检测结果")
