# 恶意样本分析系统 - FastAPI后端

基于FastAPI构建的恶意样本分析系统后端,提供恶意软件检测、ATT&CK矩阵分析、FlowViz攻击流程可视化等功能。

## 🎉 最新更新 (2026-04-29)

### 安全修复
- ✅ 修复12个Dependabot安全漏洞
- ✅ 修复26个CodeQL安全问题
- ✅ 升级易受攻击的依赖包
- ✅ 添加输入验证防止路径遍历攻击
- ✅ 修复SQL注入漏洞
- ✅ 移除日志中的敏感信息
- ✅ 修复正则表达式ReDoS漏洞
- ✅ 改进错误处理防止信息泄露

### 依赖更新
| 包名 | 旧版本 | 新版本 | 修复漏洞 |
|------|--------|--------|----------|
| xlsx | 0.17.0 | xlsx-js-style 1.2.0 | ReDoS漏洞 |
| Pillow | 10.3.0 | 12.2.0 | PSD/GZIP漏洞 |
| python-multipart | 0.0.22 | 0.0.27 | 路径遍历 |
| lxml | 5.2.1 | 6.1.0 | XXE漏洞 |
| PyJWT | 2.10.0 | 2.12.1 | crit头验证绕过 |
| aiohttp | 3.11.0 | 3.13.5 | 多个DoS漏洞 |

### 新增功能
- ✅ FlowViz AI攻击流程可视化
- ✅ 多AI提供商支持 (OpenAI, Claude)
- ✅ 流式AI分析响应
- ✅ ATT&CK技术自动映射
- ✅ 批量检测任务管理
- ✅ 实时进度反馈

### 模型说明
系统支持多种恶意软件检测模型:

#### 本地机器学习模型 (位于 `data/model/`)
- AdaBoost, Decision Tree, GBDT
- Gaussian Naive Bayes, KNN
- Logistic Regression, Random Forest
- XGBoost

#### Docker容器化模型
- 通过Docker容器部署深度学习模型
- 支持远程API调用进行推理
- 配置灵活，易于扩展

**Docker配置** (config.ini):
```ini
[docker_models]
api_base = http://192.168.8.202
timeout = 30
mode = codefender
threshold = 0.5

[codefender]
api_base = http://127.0.0.1:8001
scan_endpoint = /scan/file
predict_endpoint = /predict
timeout = 30
threshold = 0.5
```

## 项目概述

本项目是从Flask迁移到FastAPI的恶意样本分析系统,完全保留了原有功能,并进行了性能优化、架构改进和安全加固。

### 主要特性

#### 核心功能
- **用户认证**: JWT令牌认证,支持用户登录、注册、权限管理
- **ATT&CK矩阵**: MITRE ATT&CK框架可视化,技术映射管理
- **恶意软件检测**: 多引擎检测,支持文件上传和哈希查询
- **DGA检测**: 域名生成算法检测,识别恶意域名
- **域名管理**: 恶意域名库管理,支持增删改查
- **模型训练**: 机器学习模型训练和集成预测

#### FlowViz功能
- **AI分析**: 支持OpenAI和Claude AI提供商
- **流式响应**: 实时流式AI分析结果
- **攻击流程**: 自动生成攻击流程图
- **技术映射**: 自动映射MITRE ATT&CK技术
- **可视化**: 交互式攻击流程可视化
- **缓存机制**: 智能缓存提高响应速度

#### 安全特性
- **输入验证**: SHA256、task_id等参数严格验证
- **路径遍历防护**: 防止目录遍历攻击
- **SQL注入防护**: 使用参数化查询
- **命令注入防护**: 输入参数白名单验证
- **信息泄露防护**: 错误消息不包含敏感信息
- **日志安全**: 不记录API密钥等敏感数据

## 技术栈

### 后端
| 组件 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.115+ | Web框架 |
| Python | 3.9+ | 运行时 |
| MySQL | 8.0+ | 数据库 |
| PyJWT | 2.12.1+ | JWT认证 |
| aiohttp | 3.13.5+ | 异步HTTP |
| PyTorch | Latest | 深度学习 |
| scikit-learn | Latest | 机器学习 |
| XGBoost | Latest | 梯度提升 |
| LIEF | 0.14+ | PE解析 |
| yara-python | 4.3+ | YARA扫描 |

### AI集成
| 组件 | 用途 |
|------|------|
| OpenAI API | GPT-4等模型 |
| Anthropic API | Claude模型 |
| 流式响应 | 实时AI分析 |

### 前端
| 组件 | 版本 | 用途 |
|------|------|------|
| Vue.js | 3.4+ | 前端框架 |
| Element Plus | 2.5+ | UI组件 |
| ECharts | 5.4+ | 数据可视化 |
| Pinia | Latest | 状态管理 |

## 项目结构

```
fastapi/
├── app/                          # 应用主目录
│   ├── api/                      # API路由
│   │   ├── auth.py              # 用户认证API
│   │   ├── attck.py             # ATT&CK矩阵API
│   │   ├── detect.py            # 恶意软件检测API
│   │   ├── av_scan.py           # AV扫描API
│   │   ├── query.py             # 样本查询API
│   │   └── config.py            # 配置管理API
│   ├── core/                     # 核心配置
│   │   ├── config.py            # 应用配置
│   │   ├── database.py          # 数据库连接
│   │   └── security.py          # 安全工具
│   ├── schemas/                  # 数据模型
│   ├── services/                 # 业务服务
│   │   ├── av_detection/        # AV检测服务
│   │   │   ├── av_scan.py       # AV扫描
│   │   │   ├── vm_manager.py    # 虚拟机管理
│   │   │   └── yara_scan.py     # YARA扫描
│   │   └── flowviz/             # FlowViz模块
│   │       ├── config.py        # FlowViz配置
│   │       ├── providers/       # AI提供商
│   │       │   ├── openai.py    # OpenAI
│   │       │   ├── claude.py    # Claude
│   │       │   └── factory.py   # 提供商工厂
│   │       ├── routes/          # API路由
│   │       │   ├── ai.py        # AI分析
│   │       │   ├── streaming.py # 流式分析
│   │       │   └── vision.py    # 视觉分析
│   │       └── utils/           # 工具函数
│   └── utils/                    # 通用工具
├── data/                         # 数据文件
│   ├── model/                   # 训练好的模型
│   ├── features/                # 特征文件
│   └── samples/                 # 样本数据
├── feeds/                        # 机器学习分类器
│   ├── svm.py                   # SVM分类器
│   ├── knn.py                   # KNN分类器
│   ├── lstm.py                  # LSTM分类器
│   └── xgboost.py               # XGBoost分类器
├── attack/                       # ATT&CK数据
│   ├── enterprise-attack.json  # ATT&CK数据
│   └── download_parse_import.py # 数据导入脚本
├── uploads/                      # 上传文件
├── logs/                         # 日志文件
├── main.py                       # 应用入口
├── config.ini                    # 配置文件
└── requirements.txt              # 依赖列表
```

## 安装部署

### 环境要求

- Python 3.9+
- MySQL 8.0+
- Node.js 16+ (前端)

### 安装步骤

1. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置数据库**
```bash
# 编辑config.ini
[mysql]
host = localhost
port = 3306
user = root
passwd = YOUR_PASSWORD
db_category = nkrepo_category
db_family = nkrepo_family
db_platform = nkrepo_platform
```

4. **初始化数据库**
```bash
cd ../../db
python init_db.py
```

5. **启动服务**
```bash
python main.py
```

6. **访问API文档**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API端点

### 认证API
```
POST /auth/login          # 用户登录
POST /auth/register       # 用户注册
GET  /auth/userinfo       # 获取用户信息
```

### 检测API
```
POST /detect              # 文件检测
POST /detect_by_sha256    # SHA256检测
POST /batch_detect        # 批量检测
GET  /batch_status/{id}   # 批量检测状态
```

### 查询API
```
GET  /query_sha256/{sha256}     # SHA256查询
GET  /query_md5/{md5}           # MD5查询
GET  /query_category/{category} # 分类查询
GET  /query_family/{family}     # 家族查询
GET  /query_platform/{platform} # 平台查询
```

### ATT&CK API
```
GET  /dev-api/api/attck/matrix           # ATT&CK矩阵
GET  /dev-api/api/attck/techniques       # 技术列表
GET  /dev-api/api/attck/techniques/{id}  # 技术详情
GET  /dev-api/api/attck/api-components   # API组件
```

### FlowViz API
```
POST /flowviz/analyze              # AI分析
POST /flowviz/stream               # 流式分析
GET  /flowviz/providers            # 可用提供商
GET  /flowviz/health               # 健康检查
```

### AV扫描API
```
POST /av_scan                      # AV扫描
GET  /av_scan/status/{task_id}     # 扫描状态
GET  /av_scan/result/{task_id}     # 扫描结果
```

## 配置说明

### config.ini配置

```ini
[ini]
ip = 127.0.0.1
port = 8000
row_per_page = 20

[mysql]
host = localhost
port = 3306
user = root
passwd = YOUR_PASSWORD
db_category = nkrepo_category
db_family = nkrepo_family
db_platform = nkrepo_platform
charset = utf8mb4

[API]
vt_key = YOUR_VIRUSTOTAL_API_KEY

[paths]
sample_root = ../../../data/samples
web_upload_dir = ../../../data/web_upload_file
zips_dir = ../../../data/zips

[security]
secret_key = CHANGE_TO_RANDOM_SECRET
jwt_algorithm = HS256
jwt_expiration = 3600

[flowviz]
default_provider = openai
strict_mode = true
cache_enabled = true
```

### FlowViz AI配置

FlowViz支持多个AI提供商,通过环境变量配置:

```bash
# OpenAI配置
export OPENAI_API_KEY="your-openai-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4"

# Claude配置
export ANTHROPIC_API_KEY="your-claude-key"
export CLAUDE_MODEL="claude-3-5-sonnet-20241022"

# 调试模式
export FLOWVIZ_DEBUG="false"  # 生产环境必须为false
```

## 安全最佳实践

### 输入验证
```python
# SHA256验证 - 防止路径遍历
if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
    raise HTTPException(status_code=400, detail="无效的SHA256值")

# Task ID验证 - 防止目录遍历
if not re.fullmatch(r"[a-zA-Z0-9_\-]+", task_id):
    raise HTTPException(status_code=400, detail="无效的任务ID格式")
```

### SQL注入防护
```python
# 使用参数化查询
query_sql = "SELECT * FROM table WHERE id = %s"
cursor.execute(query_sql, (id,))
```

### 错误处理
```python
# 不暴露详细错误信息
except Exception as e:
    logger.error(f"详细错误: {str(e)}")  # 只记录在日志
    return {'error': '服务器内部错误'}    # 用户看到通用消息
```

## 性能优化

### 异步处理
- 使用async/await处理IO密集操作
- aiohttp进行异步HTTP请求
- 异步数据库查询

### 缓存机制
- FlowViz结果缓存
- AI响应缓存
- 配置缓存

### 数据库优化
- 256个分片表提高查询性能
- 连接池管理
- 查询优化

## 监控和日志

### 日志配置
```python
# 日志级别
INFO: 正常操作
WARNING: 警告信息
ERROR: 错误信息
DEBUG: 调试信息(生产环境禁用)
```

### 日志文件
- `app.log` - 应用日志
- `logs/` - 分类日志目录

### 敏感信息保护
- ❌ 不记录API密钥
- ❌ 不记录用户密码
- ❌ 不记录JWT令牌
- ✅ 只记录必要的调试信息

## 测试

### 运行测试
```bash
# 单元测试
pytest tests/

# 覆盖率测试
pytest --cov=app tests/

# 特定测试
pytest tests/test_detect.py -v
```

### API测试
使用Swagger UI进行交互式API测试:
- 访问 http://localhost:8000/docs
- 点击"Try it out"
- 填写参数
- 执行请求

## 故障排除

### 常见问题

**问题: 无法启动服务**
```bash
# 检查Python版本
python --version  # 需要3.9+

# 检查依赖
pip install -r requirements.txt

# 检查配置
cat config.ini
```

**问题: 数据库连接失败**
```bash
# 检查MySQL服务
systemctl status mysql

# 测试连接
mysql -u root -p

# 检查配置
grep mysql config.ini
```

**问题: FlowViz AI分析失败**
```bash
# 检查API密钥
echo $OPENAI_API_KEY

# 检查网络连接
curl https://api.openai.com/v1/models

# 查看日志
tail -f app.log | grep FlowViz
```

## 开发指南

### 添加新API端点
```python
from fastapi import APIRouter, Depends

router = APIRouter()

@router.get("/new_endpoint")
async def new_endpoint(current_user: dict = Depends(get_current_user)):
    """新端点说明"""
    # 实现逻辑
    return {"result": "success"}
```

### 添加新的AI提供商
1. 在`app/services/flowviz/providers/`创建新文件
2. 继承`BaseProvider`类
3. 实现必要的方法
4. 在`factory.py`中注册

## 贡献指南

1. Fork项目
2. 创建特性分支: `git checkout -b feature-name`
3. 提交更改: `git commit -am 'Add feature'`
4. 推送分支: `git push origin feature-name`
5. 提交Pull Request

## 许可证

NKAMG (南开大学反恶意软件研究组)

## 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 技术支持: 查看API文档 `/docs`

---

**恶意样本分析系统 FastAPI后端**
版本 2.0 | NKAMG © 2026
