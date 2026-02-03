# -*- coding: utf-8 -*-
"""
File Detection Module

Provides file-based malware detection using ensemble of ML/DL models.
"""

import sys
import logging
from pathlib import Path

# Add multi_scan directory to path
_MULTI_SCAN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MULTI_SCAN_DIR))

from ensemble_predict import run_ensemble_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def EXEDetection(file_path):
    """新版集成检测函数"""
    logger.info(f"开始检测文件: {file_path}")
    
    try:
        # 执行集成预测
        results = run_ensemble_prediction(file_path)
        
        # 格式化结果
        formatted_results = {}
        for model_name, data in results.items():
            formatted_results[model_name] = {
                "model": model_name,
                "probability": data["probability"],
                "result": data["result"]
            }
            logger.info(f"{model_name} 检测结果: {data}")
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"文件检测失败: {str(e)}")
        return {"error": str(e)}