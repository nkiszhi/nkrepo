"""
批量杀毒检测历史记录API
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from app.api.auth import get_current_user
from app.models.av_scan_history import AVBatchTaskHistory, AVScanResultHistory
from sqlalchemy import create_engine, desc, and_, or_
from sqlalchemy.orm import sessionmaker
from app.core import settings
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import quote_plus
import logging
import json

router = APIRouter()
logger = logging.getLogger(__name__)

# 创建数据库连接 - 对密码进行URL编码以处理特殊字符
encoded_password = quote_plus(settings.MYSQL_PASSWORD)
DATABASE_URL = f"mysql+pymysql://{settings.MYSQL_USER}:{encoded_password}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/av_history/list")
async def get_history_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    result_type: Optional[str] = None,
    file_name: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    获取历史记录列表
    """
    try:
        db = SessionLocal()

        # 构建查询
        query = db.query(AVBatchTaskHistory)

        # 时间筛选
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(AVBatchTaskHistory.created_at >= start_dt)
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(AVBatchTaskHistory.created_at < end_dt)

        # 结果类型筛选
        if result_type == 'malicious':
            query = query.filter(AVBatchTaskHistory.malicious_count > 0)
        elif result_type == 'safe':
            query = query.filter(AVBatchTaskHistory.malicious_count == 0)

        # 文件名搜索
        if file_name:
            # 需要关联结果表搜索
            task_ids = db.query(AVScanResultHistory.task_id).filter(
                AVScanResultHistory.file_name.like(f"%{file_name}%")
            ).all()
            task_ids = [t[0] for t in task_ids]
            if task_ids:
                query = query.filter(AVBatchTaskHistory.task_id.in_(task_ids))
            else:
                # 没有匹配的结果
                db.close()
                return {
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "items": []
                }

        # 按创建时间倒序
        query = query.order_by(desc(AVBatchTaskHistory.created_at))

        # 总数
        total = query.count()

        # 分页
        offset = (page - 1) * page_size
        items = query.offset(offset).limit(page_size).all()

        # 格式化结果
        result_items = []
        for item in items:
            result_items.append({
                "id": item.id,
                "task_id": item.task_id,
                "user_id": item.user_id,
                "total_files": item.total_files,
                "selected_engines": item.selected_engines,
                "malicious_count": item.malicious_count,
                "safe_count": item.safe_count,
                "status": item.status,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "completed_at": item.completed_at.isoformat() if item.completed_at else None
            })

        db.close()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": result_items
        }

    except Exception as e:
        logger.error(f"获取历史记录列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


@router.get("/av_history/detail/{task_id}")
async def get_history_detail(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    获取历史记录详情
    """
    try:
        db = SessionLocal()

        # 查询任务信息
        task = db.query(AVBatchTaskHistory).filter(
            AVBatchTaskHistory.task_id == task_id
        ).first()

        if not task:
            db.close()
            raise HTTPException(status_code=404, detail="任务不存在")

        # 查询检测结果
        results = db.query(AVScanResultHistory).filter(
            AVScanResultHistory.task_id == task_id
        ).all()

        # 格式化结果
        result_list = []
        for r in results:
            result_list.append({
                "file_name": r.file_name,
                "file_size": r.file_size,
                "engines": r.engine_results,
                "malicious_count": r.malicious_count,
                "safe_count": r.safe_count,
                "tag": r.tag,
                "tag_type": r.tag_type
            })

        response = {
            "task_id": task.task_id,
            "status": task.status,
            "total_files": task.total_files,
            "selected_engines": task.selected_engines,
            "malicious_count": task.malicious_count,
            "safe_count": task.safe_count,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "results": result_list
        }

        db.close()
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取历史记录详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取详情失败: {str(e)}")


@router.post("/av_history/tag")
async def add_or_update_tag(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """
    添加或更新标签
    """
    try:
        task_id = request.get('task_id')
        file_name = request.get('file_name')
        tag = request.get('tag')
        tag_type = request.get('tag_type', 'custom')

        if not task_id or not file_name or not tag:
            raise HTTPException(status_code=400, detail="缺少必要参数")

        db = SessionLocal()

        # 查询结果记录
        result = db.query(AVScanResultHistory).filter(
            and_(
                AVScanResultHistory.task_id == task_id,
                AVScanResultHistory.file_name == file_name
            )
        ).first()

        if not result:
            db.close()
            raise HTTPException(status_code=404, detail="记录不存在")

        # 更新标签
        result.tag = tag
        result.tag_type = tag_type
        result.tag_updated_at = datetime.now()

        db.commit()
        db.close()

        return {
            "success": True,
            "message": "标签已更新"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新标签失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新标签失败: {str(e)}")


@router.delete("/av_history/{task_id}")
async def delete_history(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    删除历史记录
    """
    try:
        db = SessionLocal()

        # 删除检测结果
        db.query(AVScanResultHistory).filter(
            AVScanResultHistory.task_id == task_id
        ).delete()

        # 删除任务记录
        deleted = db.query(AVBatchTaskHistory).filter(
            AVBatchTaskHistory.task_id == task_id
        ).delete()

        if deleted == 0:
            db.rollback()
            db.close()
            raise HTTPException(status_code=404, detail="记录不存在")

        db.commit()
        db.close()

        return {
            "success": True,
            "message": "记录已删除"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除历史记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


def save_scan_to_history(
    task_id: str,
    user_id: int,
    status: str,
    total_files: int,
    selected_engines: List[str],
    scan_results: List[dict]
):
    """
    保存扫描结果到历史记录（内部函数）
    """
    try:
        db = SessionLocal()

        # 计算统计信息
        malicious_count = sum(1 for r in scan_results if r.get('malicious_count', 0) > 0)
        safe_count = total_files - malicious_count

        # 创建任务历史记录
        task_history = AVBatchTaskHistory(
            task_id=task_id,
            user_id=user_id,
            status=status,
            total_files=total_files,
            selected_engines=selected_engines,
            malicious_count=malicious_count,
            safe_count=safe_count,
            created_at=datetime.now(),
            completed_at=datetime.now() if status == 'completed' else None
        )
        db.add(task_history)

        # 保存每个文件的检测结果
        for result in scan_results:
            result_history = AVScanResultHistory(
                task_id=task_id,
                file_name=result.get('file_name', ''),
                file_size=result.get('file_size', ''),
                engine_results=result.get('engines', {}),
                malicious_count=result.get('malicious_count', 0),
                safe_count=result.get('safe_count', 0),
                created_at=datetime.now()
            )
            db.add(result_history)

        db.commit()
        db.close()

        logger.info(f"历史记录保存成功: task_id={task_id}")
        return True

    except Exception as e:
        logger.error(f"保存历史记录失败: {str(e)}")
        return False
