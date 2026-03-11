"""
ATT&CK相关的Pydantic模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class MatrixResponse(BaseModel):
    """矩阵响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="矩阵数据")
    message: Optional[str] = Field(None, description="消息")


class MatrixStatsResponse(BaseModel):
    """矩阵统计响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="统计数据")


class FunctionListResponse(BaseModel):
    """函数列表响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="函数列表")
    pagination: Optional[Dict[str, Any]] = Field(None, description="分页信息")


class FunctionDetailResponse(BaseModel):
    """函数详情响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="函数详情")


class TechniqueListResponse(BaseModel):
    """技术列表响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="技术列表")
    pagination: Optional[Dict[str, Any]] = Field(None, description="分页信息")


class TechniqueDetailResponse(BaseModel):
    """技术详情响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="技术详情")


class ApiComponentListResponse(BaseModel):
    """API组件列表响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="API组件列表")
    pagination: Optional[Dict[str, Any]] = Field(None, description="分页信息")


class ApiComponentDetailResponse(BaseModel):
    """API组件详情响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="API组件详情")


class TechniqueMappingResponse(BaseModel):
    """技术映射响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="映射数据")
    pagination: Optional[Dict[str, Any]] = Field(None, description="分页信息")
