"""
FlowViz服务
整合flowviz模块的功能
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional

# 添加flowviz到路径
flowviz_path = os.path.join(os.path.dirname(__file__), 'flowviz')
if flowviz_path not in sys.path:
    sys.path.insert(0, flowviz_path)

logger = logging.getLogger(__name__)


class FlowVizService:
    """FlowViz服务类"""
    
    def __init__(self):
        self.providers = {}
        self.config = None
        self._load_config()
        self._load_providers()
    
    def _load_config(self):
        """加载FlowViz配置"""
        try:
            from app.services.flowviz.config import FlowVizConfig
            self.config = FlowVizConfig()
            logger.info("FlowViz配置加载成功")
        except Exception as e:
            logger.warning(f"FlowViz配置加载失败: {str(e)}")
            self.config = None
    
    def _load_providers(self):
        """加载可用的Provider"""
        try:
            # 尝试导入flowviz的provider
            from app.services.flowviz.providers.factory import ProviderFactory
            self.provider_factory = ProviderFactory()
            logger.info("FlowViz Provider加载成功")
        except Exception as e:
            logger.warning(f"FlowViz Provider加载失败: {str(e)}")
            self.provider_factory = None
    
    def get_providers(self) -> List[Dict[str, Any]]:
        """获取可用的Provider列表"""
        # 基础Provider列表
        providers = [
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "OpenAI GPT模型",
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                "available": True
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "description": "Claude模型",
                "models": ["claude-2", "claude-3-opus", "claude-3-sonnet"],
                "available": True
            },
            {
                "id": "deepseek",
                "name": "DeepSeek",
                "description": "DeepSeek模型",
                "models": ["deepseek-chat", "deepseek-coder"],
                "available": True
            },
            {
                "id": "local",
                "name": "本地模型",
                "description": "本地部署的模型",
                "models": ["local-model"],
                "available": True
            }
        ]
        
        # 如果成功加载了provider factory,更新可用状态
        if self.provider_factory:
            try:
                available_providers = self.provider_factory.get_available_providers()
                for provider in providers:
                    provider['available'] = provider['id'] in available_providers
            except Exception as e:
                logger.warning(f"获取可用Provider失败: {str(e)}")
        
        return providers
    
    async def analyze(
        self,
        input_type: str,
        input_value: str,
        provider: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        分析攻击流
        
        Args:
            input_type: 输入类型 (url, text, code等)
            input_value: 输入值
            provider: Provider ID
            options: 其他选项
        
        Returns:
            分析结果,包含nodes和edges
        """
        try:
            # 如果有真实的provider factory,使用它
            if self.provider_factory and provider:
                try:
                    provider_instance = self.provider_factory.get_provider(provider)
                    result = await provider_instance.analyze(input_value, **(options or {}))
                    return result
                except Exception as e:
                    logger.warning(f"使用Provider {provider} 分析失败: {str(e)}")
            
            # 尝试使用AI路由
            try:
                from app.services.flowviz.routes.ai import analyze_with_ai
                result = await analyze_with_ai(
                    input_type=input_type,
                    input_value=input_value,
                    provider=provider,
                    **(options or {})
                )
                return result
            except Exception as e:
                logger.warning(f"AI分析失败: {str(e)}")
            
            # 否则返回模拟数据
            return self._generate_mock_analysis(input_type, input_value)
            
        except Exception as e:
            logger.error(f"分析失败: {str(e)}")
            raise
    
    def _generate_mock_analysis(self, input_type: str, input_value: str) -> Dict[str, Any]:
        """生成模拟分析结果"""
        import time
        
        # 根据输入类型生成不同的节点
        if input_type == "url":
            nodes = [
                {
                    "id": "node1",
                    "type": "input",
                    "data": {"label": f"URL: {input_value[:30]}..."},
                    "position": {"x": 100, "y": 100}
                },
                {
                    "id": "node2",
                    "type": "process",
                    "data": {"label": "DNS解析"},
                    "position": {"x": 300, "y": 100}
                },
                {
                    "id": "node3",
                    "type": "process",
                    "data": {"label": "HTTP请求"},
                    "position": {"x": 300, "y": 200}
                },
                {
                    "id": "node4",
                    "type": "output",
                    "data": {"label": "响应分析"},
                    "position": {"x": 500, "y": 150}
                }
            ]
            edges = [
                {"id": "edge1", "source": "node1", "target": "node2"},
                {"id": "edge2", "source": "node1", "target": "node3"},
                {"id": "edge3", "source": "node2", "target": "node4"},
                {"id": "edge4", "source": "node3", "target": "node4"}
            ]
        elif input_type == "code":
            nodes = [
                {
                    "id": "node1",
                    "type": "input",
                    "data": {"label": "代码输入"},
                    "position": {"x": 100, "y": 100}
                },
                {
                    "id": "node2",
                    "type": "process",
                    "data": {"label": "语法分析"},
                    "position": {"x": 300, "y": 100}
                },
                {
                    "id": "node3",
                    "type": "process",
                    "data": {"label": "行为提取"},
                    "position": {"x": 300, "y": 200}
                },
                {
                    "id": "node4",
                    "type": "output",
                    "data": {"label": "攻击链"},
                    "position": {"x": 500, "y": 150}
                }
            ]
            edges = [
                {"id": "edge1", "source": "node1", "target": "node2"},
                {"id": "edge2", "source": "node2", "target": "node3"},
                {"id": "edge3", "source": "node3", "target": "node4"}
            ]
        else:
            nodes = [
                {
                    "id": "node1",
                    "type": "input",
                    "data": {"label": input_value[:50] + "..." if len(input_value) > 50 else input_value},
                    "position": {"x": 100, "y": 100}
                },
                {
                    "id": "node2",
                    "type": "process",
                    "data": {"label": "分析处理"},
                    "position": {"x": 300, "y": 100}
                },
                {
                    "id": "node3",
                    "type": "output",
                    "data": {"label": "结果输出"},
                    "position": {"x": 500, "y": 100}
                }
            ]
            edges = [
                {"id": "edge1", "source": "node1", "target": "node2"},
                {"id": "edge2", "source": "node2", "target": "node3"}
            ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "analysis_time": 0.5,
            "provider": "mock"
        }
    
    def save_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """保存分析结果"""
        import json
        import time
        
        try:
            # 生成ID
            analysis_id = f"analysis_{int(time.time())}"
            analysis_data['id'] = analysis_id
            analysis_data['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存到文件
            save_dir = os.path.join(os.path.dirname(__file__), 'flowviz/data')
            os.makedirs(save_dir, exist_ok=True)
            
            history_file = os.path.join(save_dir, 'history.json')
            
            # 加载现有历史
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # 添加新记录
            history.insert(0, analysis_data)
            
            # 保存
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析结果已保存: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}")
            raise
    
    def get_history(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """获取分析历史"""
        import json
        
        try:
            history_file = os.path.join(os.path.dirname(__file__), 'flowviz/data/history.json')
            
            if not os.path.exists(history_file):
                return {
                    'data': [],
                    'total': 0,
                    'page': page,
                    'page_size': page_size
                }
            
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # 分页
            start = (page - 1) * page_size
            end = start + page_size
            page_data = history[start:end]
            
            return {
                'data': page_data,
                'total': len(history),
                'page': page,
                'page_size': page_size
            }
            
        except Exception as e:
            logger.error(f"获取历史记录失败: {str(e)}")
            return {
                'data': [],
                'total': 0,
                'page': page,
                'page_size': page_size
            }
    
    async def stream_analyze(
        self,
        input_type: str,
        input_value: str,
        provider: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        流式分析(支持SSE)
        
        Yields:
            分析过程的流式数据
        """
        try:
            # 尝试使用流式路由
            from app.services.flowviz.routes.streaming import stream_analyze as stream_func
            async for chunk in stream_func(
                input_type=input_type,
                input_value=input_value,
                provider=provider,
                **(options or {})
            ):
                yield chunk
        except Exception as e:
            logger.warning(f"流式分析失败: {str(e)}")
            # 返回简单的流式数据
            import json
            yield f"data: {json.dumps({'status': 'start'})}\n\n"
            yield f"data: {json.dumps({'status': 'processing', 'progress': 50})}\n\n"
            result = self._generate_mock_analysis(input_type, input_value)
            yield f"data: {json.dumps({'status': 'complete', 'result': result})}\n\n"


# 创建全局服务实例
flowviz_service = FlowVizService()
