"""
批量杀毒检测历史记录数据模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class AVBatchTaskHistory(Base):
    """批量检测任务历史表"""
    __tablename__ = "av_batch_task_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)

    # 任务信息
    status = Column(String(20), nullable=False)  # pending/running/completed/failed
    total_files = Column(Integer, nullable=False)
    selected_engines = Column(JSON, nullable=False)  # 选择的引擎列表

    # 统计信息
    malicious_count = Column(Integer, default=0)
    safe_count = Column(Integer, default=0)

    # 时间信息
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    completed_at = Column(DateTime)

    # 错误信息
    error_message = Column(Text)

    # 添加复合索引
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
    )


class AVScanResultHistory(Base):
    """检测结果历史表"""
    __tablename__ = "av_scan_result_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), nullable=False, index=True)

    # 文件信息
    file_name = Column(String(255), nullable=False)
    file_size = Column(String(50))

    # 检测结果
    engine_results = Column(JSON, nullable=False)  # {"Avira": "malicious", "McAfee": "safe"}
    malicious_count = Column(Integer, default=0)
    safe_count = Column(Integer, default=0)

    # 标签信息
    tag = Column(String(50))
    tag_type = Column(String(20))  # predefined/custom

    # 时间信息
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    tag_updated_at = Column(DateTime)

    # 添加复合索引
    __table_args__ = (
        Index('idx_task_file', 'task_id', 'file_name'),
    )
