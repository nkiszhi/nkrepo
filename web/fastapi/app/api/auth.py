"""
认证相关API - 完全按照旧Flask实现
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.schemas.auth import LoginRequest, LoginResponse
from app.services.auth_service import AuthService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
security = HTTPBearer()

# 创建认证服务实例
auth_service = AuthService()


@router.post("/login")
async def login(request: LoginRequest):
    """
    用户登录 - 完全按照旧Flask的/api/login实现
    """
    try:
        logger.info(f"登录请求: username={request.username}")
        
        # 认证用户 - 完全按照旧Flask逻辑
        result = auth_service.authenticate_user(request.username, request.password)
        
        if not result['success']:
            logger.warning(f"认证失败: {result['message']}")
            raise HTTPException(status_code=401, detail=result['message'])
        
        logger.info(f"登录成功: username={request.username}")
        
        # 返回格式完全按照旧Flask
        return {
            'success': True,
            'token': result['token'],
            'username': result['username'],
            'message': result['message']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    获取当前用户(依赖注入)
    用于需要认证的API
    """
    token = credentials.credentials
    payload = auth_service.verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


@router.get("/verify")
async def verify_token(current_user: dict = Depends(get_current_user)):
    """
    验证令牌有效性
    """
    return {
        "success": True,
        "message": "令牌有效",
        "user": current_user
    }
