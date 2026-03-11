# flowviz/__init__.py
"""
FlowViz 模块主文件 (FastAPI版本)
"""
import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# FlowViz模块已整合到FastAPI
# 不再使用Flask Blueprint

__all__ = ['current_dir']
