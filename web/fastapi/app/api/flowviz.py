"""
FlowViz攻击流可视化API
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from app.api.auth import get_current_user
from app.schemas.flowviz import FlowVizResponse, ProviderResponse, AnalysisRequest, AnalysisResponse
from app.services.flowviz_service import flowviz_service
import logging
import os

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
@router.get("", response_class=HTMLResponse)
async def flowviz_root():
    """
    FlowViz主页
    """
    try:
        index_file = os.path.join(FLOWVIZ_DIR, 'index.html')
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return """
            <html>
                <head><title>FlowViz</title></head>
                <body>
                    <h1>FlowViz - 攻击流可视化</h1>
                    <p>FlowViz服务正在运行</p>
                    <p>API文档: <a href="/docs">/docs</a></p>
                </body>
            </html>
            """
    except Exception as e:
        logger.error(f"加载FlowViz主页失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def flowviz_health():
    """
    FlowViz健康检查
    """
    return {
        "status": "healthy",
        "service": "flowviz",
        "version": "2.0.0"
    }


@router.get("/api/providers", response_model=ProviderResponse)
async def get_providers(current_user: dict = Depends(get_current_user)):
    """
    获取可用的Provider列表
    """
    try:
        providers = flowviz_service.get_providers()
        
        return ProviderResponse(
            success=True,
            providers=providers
        )
        
    except Exception as e:
        logger.error(f"获取Provider列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_flow(
    request: AnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    分析攻击流
    """
    try:
        result = await flowviz_service.analyze(
            input_type=request.input_type,
            input_value=request.input_value,
            provider=request.provider,
            options=request.options
        )
        
        return AnalysisResponse(
            success=True,
            message="分析完成",
            nodes=result.get('nodes', []),
            edges=result.get('edges', []),
            analysis_time=result.get('analysis_time', 0)
        )
        
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/history")
async def get_analysis_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """
    获取分析历史
    """
    try:
        # 从数据库或文件加载历史记录
        history_file = os.path.join(FLOWVIZ_DIR, 'history.json')
        
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # 分页
        start = (page - 1) * page_size
        end = start + page_size
        page_data = history[start:end]
        
        return {
            "success": True,
            "data": page_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": len(history),
                "total_pages": (len(history) + page_size - 1) // page_size
            }
        }
        
    except Exception as e:
        logger.error(f"获取历史记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/save")
async def save_analysis(
    analysis_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """
    保存分析结果
    """
    try:
        history_file = os.path.join(FLOWVIZ_DIR, 'history.json')
        
        # 加载现有历史
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # 添加新记录
        import time
        analysis_data['id'] = f"analysis_{int(time.time())}"
        analysis_data['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        history.insert(0, analysis_data)
        
        # 保存
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "message": "保存成功",
            "id": analysis_data['id']
        }
        
    except Exception as e:
        logger.error(f"保存分析结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
