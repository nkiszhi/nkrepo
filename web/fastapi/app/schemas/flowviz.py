"""
FlowViz相关的Pydantic模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class FlowVizResponse(BaseModel):
    """FlowViz响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    data: Optional[Dict[str, Any]] = Field(None, description="数据")


class ProviderResponse(BaseModel):
    """Provider响应"""
    success: bool = Field(..., description="是否成功")
    providers: Optional[List[Dict[str, Any]]] = Field(None, description="Provider列表")


class AnalysisRequest(BaseModel):
    """分析请求"""
    input_type: str = Field(..., description="输入类型")
    input_value: str = Field(..., description="输入值")
    provider: Optional[str] = Field(None, description="Provider")
    options: Optional[Dict[str, Any]] = Field(None, description="选项")


class AnalysisResponse(BaseModel):
    """分析响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    nodes: Optional[List[Dict[str, Any]]] = Field(None, description="节点列表")
    edges: Optional[List[Dict[str, Any]]] = Field(None, description="边列表")
    analysis_time: Optional[float] = Field(None, description="分析时间")
