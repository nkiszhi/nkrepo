# flowviz/routes/__init__.py
"""
FlowViz路由模块
确保所有蓝图都被正确导出
"""

# 导入所有蓝图模块
from . import ai
from . import streaming
from . import fetch
from . import providers
from . import vision

# 导出所有蓝图对象
__all__ = ['ai', 'streaming', 'fetch', 'providers', 'vision']

# 验证蓝图存在
def validate_blueprints():
    """验证所有蓝图都已正确定义"""
    blueprints = {
        'ai': getattr(ai, 'bp', None),
        'streaming': getattr(streaming, 'bp', None),
        'fetch': getattr(fetch, 'bp', None),
        'providers': getattr(providers, 'bp', None),
        'vision': getattr(vision, 'bp', None)
    }
    
    for name, bp in blueprints.items():
        if bp is None:
            print(f"⚠️ 警告: {name}模块没有定义'bp'蓝图对象")
        else:
            print(f"✅ {name}模块蓝图已定义")
    
    return all(bp is not None for bp in blueprints.values())

# 在导入时验证
if __name__ != "__main__":
    validate_blueprints()