"""
恶意样本分析系统 - FastAPI主应用
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core import settings
from app.api import (
    auth_router,
    detect_router,
    attck_router,
    flowviz_router,
    query_router,
    config_router,
    flowviz_streaming_router,
    av_scan_router
)

# 导入vue_data生成函数
from app.services.data.vue_data_service import generate_frontend_data

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def start_vue_data_scheduler():
    """启动vue_data定时任务 - 完全按照旧Flask实现"""
    logger.info("启动vue_data后台生成任务")
    
    # 启动时先运行一次
    try:
        logger.info("执行首次vue_data生成...")
        success = generate_frontend_data()
        if success:
            logger.info("首次vue_data生成成功")
        else:
            logger.error("首次vue_data生成失败")
    except Exception as e:
        logger.error(f"首次vue_data生成异常: {str(e)}")
    
    # 设置定时任务
    scheduler = BackgroundScheduler()
    
    # 每天凌晨2点运行
    scheduler.add_job(
        generate_frontend_data,
        CronTrigger(hour=2, minute=0),
        id='daily_vue_data_generation',
        replace_existing=True
    )
    
    # 每6小时运行一次(测试用)
    scheduler.add_job(
        generate_frontend_data,
        'interval',
        hours=6,
        id='interval_vue_data_generation',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("vue_data定时任务已启动:每天凌晨2点运行,每6小时运行一次(测试)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info(f"[启动] {settings.APP_NAME} v{settings.APP_VERSION} 启动中...")
    
    # 创建必要的目录
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # 更新Vue前端配置
    await update_vue_config()
    
    # 启动vue_data定时任务
    start_vue_data_scheduler()
    
    yield
    
    # 关闭时执行
    logger.info(f"[关闭] {settings.APP_NAME} 关闭中...")


async def update_vue_config():
    """更新Vue前端配置"""
    try:
        import socket
        # 获取正确的网络IP地址
        def get_network_ip():
            """获取本机的网络IP地址"""
            try:
                # 创建一个UDP socket连接到外部地址(不会真正发送数据)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # 连接到Google DNS服务器(不会真正发送数据)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except Exception:
                # 如果失败,尝试使用hostname获取
                return socket.gethostbyname(socket.gethostname())
        
        actual_ip = get_network_ip()
        base_url = f"http://{actual_ip}:{settings.PORT}"
        
        vue_dir = os.path.join(os.path.dirname(__file__), settings.VUE_DIR)
        
        # 1. 更新.env文件
        env_files = ['.env.development', '.env.production']
        for env_file in env_files:
            env_path = os.path.join(vue_dir, env_file)
            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            
            lines = []
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            updated = False
            new_lines = []
            for line in lines:
                if line.strip().startswith('VITE_APP_BASE_API'):
                    new_lines.append(f"VITE_APP_BASE_API = '{base_url}'\n")
                    updated = True
                else:
                    new_lines.append(line)
            
            if not updated:
                if new_lines and not new_lines[-1].endswith('\n'):
                    new_lines.append('\n')
                new_lines.append(f"VITE_APP_BASE_API = '{base_url}'\n")
            
            with open(env_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            logger.info(f"[OK] Vue配置已更新: {env_file}, API地址: {base_url}")
        
        # 2. 更新public/config.ini
        config_ini_path = os.path.join(vue_dir, 'public/config.ini')
        config_content = f"""# Vue前端API配置
# 后端启动时会自动更新此文件

[api]
baseUrl = {base_url}
prefix = /api
"""
        with open(config_ini_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        logger.info(f"[OK] Vue config.ini已更新: {config_ini_path}")
        
    except Exception as e:
        logger.error(f"[ERROR] 更新Vue配置失败: {str(e)}")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于FastAPI的恶意样本分析系统后端",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
if os.path.exists(settings.UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# 注册路由 - 完全兼容旧后端路由
app.include_router(auth_router, prefix="/api", tags=["认证"])
app.include_router(detect_router, tags=["检测"])
app.include_router(attck_router, tags=["ATT&CK"])
app.include_router(flowviz_router, prefix="/flowviz", tags=["攻击流可视化"])
app.include_router(flowviz_streaming_router, prefix="/flowviz", tags=["FlowViz流式分析"])
app.include_router(query_router, tags=["查询"])
app.include_router(config_router, prefix="/api", tags=["配置"])
app.include_router(av_scan_router, prefix="/api", tags=["分布式杀毒扫描"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": f"欢迎使用{settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
