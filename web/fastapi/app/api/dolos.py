"""
Dolos代码同源性检测API路由
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import shutil
import os
from pathlib import Path
from datetime import datetime

from ..core import settings
from ..services.dolos_service import DolosAnalyzer

router = APIRouter()
analyzer = DolosAnalyzer()


@router.post("/analyze")
async def analyze_code(files: List[UploadFile] = File(...)):
    """
    分析代码文件相似度
    
    Args:
        files: 上传的代码文件列表（至少2个文件）
    
    Returns:
        分析结果，包含文件对之间的相似度信息
    """
    try:
        # 验证文件数量
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="至少需要上传2个文件进行分析")
        
        # 创建临时目录
        temp_dir = Path(settings.UPLOAD_DIR) / "dolos_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存上传的文件
        file_paths = []
        for file in files:
            # 验证文件类型
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in settings.DOLOS_SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"不支持的文件类型: {file_ext}"
                )
            
            # 保存文件
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(str(file_path))
        
        # 执行分析
        result = await analyzer.analyze(file_paths)
        
        # 清理临时文件
        for path in file_paths:
            try:
                os.remove(path)
            except Exception:
                pass
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@router.get("/history")
async def get_analysis_history(
    skip: int = Query(0, ge=0, description="跳过记录数"),
    limit: int = Query(10, ge=1, le=100, description="返回记录数"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期")
):
    """
    获取分析历史记录
    
    Args:
        skip: 跳过的记录数
        limit: 返回的记录数
        search: 搜索关键词（文件名或分析ID）
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
    
    Returns:
        历史记录列表
    """
    try:
        result = await analyzer.get_history(
            skip=skip,
            limit=limit,
            search=search,
            start_date=start_date,
            end_date=end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


@router.get("/result/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """
    获取特定分析结果
    
    Args:
        analysis_id: 分析ID
    
    Returns:
        分析结果详情
    """
    try:
        result = await analyzer.get_result(analysis_id)
        if not result:
            raise HTTPException(status_code=404, detail="分析结果不存在")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取分析结果失败: {str(e)}")


@router.delete("/result/{analysis_id}")
async def delete_analysis_result(analysis_id: str):
    """
    删除分析结果
    
    Args:
        analysis_id: 分析ID
    
    Returns:
        删除结果
    """
    try:
        success = await analyzer.delete_result(analysis_id)
        if not success:
            raise HTTPException(status_code=404, detail="分析结果不存在")
        return {"message": "删除成功", "analysis_id": analysis_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/file-types")
async def get_supported_file_types():
    """
    获取支持的文件类型列表
    
    Returns:
        支持的文件类型列表
    """
    return {
        "extensions": settings.DOLOS_SUPPORTED_EXTENSIONS,
        "languages": {
            ".py": "Python",
            ".js": "JavaScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".ts": "TypeScript",
            ".go": "Go",
            ".rs": "Rust",
            ".php": "PHP",
            ".rb": "Ruby"
        }
    }


@router.post("/analyze-urls")
async def analyze_from_urls(urls: List[str]):
    """
    从URL批量分析代码文件
    
    Args:
        urls: 文件URL列表
    
    Returns:
        分析结果
    """
    try:
        if len(urls) < 2:
            raise HTTPException(status_code=400, detail="至少需要2个URL进行分析")
        
        result = await analyzer.analyze_from_urls(urls)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")
