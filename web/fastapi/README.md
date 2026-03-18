# 恶意样本分析系统 - FastAPI后端

基于FastAPI构建的恶意样本分析系统后端,提供恶意软件检测、分布式杀毒扫描、ATT&CK矩阵分析、FlowViz攻击流程可视化等功能。

## 项目概述

本项目是从Flask迁移到FastAPI的恶意样本分析系统,完全保留了原有功能,并进行了性能优化和架构改进,新增了分布式杀毒扫描功能。

### 主要特性

- **用户认证**: JWT令牌认证,支持用户登录、注册、权限管理
- **样本检测**: 多引擎检测,支持文件上传和哈希查询,集成VirusTotal检测
- **分布式杀毒扫描**: 支持多虚拟机分布式杀毒扫描,单文件和批量扫描
- **ATT&CK矩阵**: MITRE ATT&CK框架可视化,技术映射管理,API组件映射
- **DGA检测**: 域名生成算法检测,识别恶意域名,支持多种机器学习算法
- **域名管理**: 恶意域名库管理,支持增删改查
- **模型训练**: 机器学习模型训练和集成预测,支持多种分类器
- **FlowViz分析**: 攻击流程可视化,支持流式AI分析,支持多种AI提供商
- **配置管理**: 系统配置管理,支持动态更新
- **定时任务**: 自动生成前端数据,定时更新统计信息
- **数据查询**: 样本查询,域名查询,分类查询

## 技术栈

### 后端框架
- **FastAPI 0.109.0**: 现代化、高性能的Web框架
- **Uvicorn**: ASGI服务器
- **Pydantic**: 数据验证和设置管理

### 数据库
- **MySQL 8.0**: 关系型数据库
- **PyMySQL**: MySQL驱动
- **SQLAlchemy**: ORM框架

### 认证与安全
- **JWT (PyJWT 2.8.0)**: 令牌认证
- **bcrypt**: 密码加密

### AI集成
- **OpenAI API**: GPT-4o模型
- **Anthropic API**: Claude模型
- **aiohttp**: 异步HTTP请求

### 异步与调度
- **asyncio**: 异步编程
- **APScheduler 3.10.0**: 定时任务调度

### 机器学习
- **scikit-learn**: 机器学习算法
- **scipy**: 科学计算
- **numpy**: 数值计算
- **pandas**: 数据处理

### 数据处理
- **pandas**: 数据分析
- **numpy**: 数值计算

## 项目结构

```
web/fastapi/
├── app/                          # 应用主目录
│   ├── api/                      # API路由
│   │   ├── auth.py              # 用户认证API
│   │   ├── attck.py             # ATT&CK矩阵API
│   │   ├── detect.py            # 恶意软件检测API
│   │   ├── flowviz.py           # FlowViz基础API
│   │   ├── flowviz_streaming.py # FlowViz流式分析API
│   │   ├── query.py             # 查询API
│   │   └── config.py            # 配置管理API
│   ├── core/                     # 核心配置
│   │   ├── config.py            # 应用配置
│   │   └── database.py          # 数据库连接
│   ├── schemas/                  # 数据模型
│   │   ├── auth.py              # 认证数据模型
│   │   ├── attck.py             # ATT&CK数据模型
│   │   ├── detect.py            # 检测数据模型
│   │   ├── flowviz.py           # FlowViz数据模型
│   │   └── query.py             # 查询数据模型
│   ├── scripts/                  # 业务脚本
│   │   ├── config_manager.py    # 配置管理
│   │   ├── dname_add.py         # 域名添加
│   │   ├── dname_del.py         # 域名删除
│   │   ├── dname_query.py       # 域名查询
│   │   ├── init_config.py       # 初始化配置
│   │   └── train_model.py       # 模型训练
│   ├── services/                 # 业务服务
│   │   ├── auth_service.py      # 认证服务
│   │   ├── flowviz_service.py   # FlowViz服务
│   │   ├── data/                # 数据服务
│   │   │   ├── vue_data.py      # Vue数据生成
│   │   │   └── vue_data_service.py # Vue数据服务
│   │   ├── detection/           # 检测服务
│   │   │   ├── dga_detection.py # DGA检测
│   │   │   ├── file_detect.py   # 文件检测
│   │   │   └── ensemble_predict.py # 集成预测
│   │   ├── external/            # 外部服务
│   │   │   └── api_vt.py        # VirusTotal API
│   │   └── flowviz/             # FlowViz模块
│   │       ├── config.py        # FlowViz配置
│   │       ├── providers/       # AI提供商
│   │       │   ├── base.py      # 基础提供商
│   │       │   ├── openai.py    # OpenAI提供商
│   │       │   ├── claude.py    # Claude提供商
│   │       │   └── factory.py   # 提供商工厂
│   │       ├── routes/          # FlowViz路由
│   │       │   ├── ai.py        # AI分析路由
│   │       │   ├── fetch.py     # 数据获取路由
│   │       │   ├── providers.py # 提供商路由
│   │       │   ├── streaming.py # 流式传输路由
│   │       │   └── vision.py    # 视觉分析路由
│   │       └── utils/           # FlowViz工具
│   │           ├── advanced_parser.py # 高级解析器
│   │           ├── logger.py    # 日志工具
│   │           ├── security.py  # 安全工具
│   │           ├── sse.py       # SSE工具
│   │           ├── stream_parser.py # 流式解析器
│   │           ├── strict_validator.py # 严格验证器
│   │           └── technical_processor.py # 技术处理器
│   └── utils/                    # 工具函数
│       ├── feature_extraction.py # 特征提取
│       └── flask_mysql.py       # MySQL工具
├── data/                         # 数据文件
│   ├── model/                   # 训练好的模型
│   ├── features/                # 特征文件
│   ├── dga_date.csv             # DGA日期数据
│   ├── gib_model.pki            # GIB模型
│   ├── hmm_matrix.csv           # HMM矩阵
│   ├── lstm_score_rank.csv      # LSTM评分排名
│   ├── n_gram_rank_freq.txt     # N-gram排名频率
│   ├── private_tld.txt          # 私有顶级域名
│   ├── tld.txt                  # 顶级域名
│   ├── tld_rank.txt             # 顶级域名排名
│   └── vs_date.csv              # VS日期数据
├── feeds/                        # 机器学习分类器
│   ├── adaboost.py              # AdaBoost分类器
│   ├── danalysis.py             # 域名分析
│   ├── decisiontree.py          # 决策树分类器
│   ├── gbdt.py                  # GBDT分类器
│   ├── knn.py                   # KNN分类器
│   ├── logisticregression.py    # 逻辑回归分类器
│   ├── lstm.py                  # LSTM分类器
│   ├── naivebayes.py            # 朴素贝叶斯分类器
│   ├── pvalue.py                # P值计算
│   ├── randomforest.py          # 随机森林分类器
│   ├── svm.py                   # SVM分类器
│   └── xgboost.py               # XGBoost分类器
├── attack/                       # ATT&CK数据
│   ├── enterprise-attack.json   # ATT&CK数据
│   ├── tactic_translations.json # 战术翻译
│   └── download_parse_import.py # 数据导入脚本
├── uploads/                      # 上传文件目录
├── logs/                         # 日志文件目录
├── main.py                       # 应用入口
├── config.ini                    # 配置文件(不提交)
├── config.ini.example            # 配置文件示例
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

## 安装部署

### 环境要求

- Python 3.9+
- MySQL 8.0+
- Node.js 16+ (前端)
- pip

### 安装步骤

1. **克隆项目**
```bash
git clone <repository_url>
cd nkrepo/web/fastapi
```

2. **创建虚拟环境**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置数据库**

复制配置文件示例:
```bash
cp config.ini.example config.ini
```

编辑 `config.ini` 文件,配置数据库连接:
```ini
[mysql]
host = 127.0.0.1
port = 3306
user = your_mysql_user
passwd = your_mysql_password
db = malicious
db_web = webdatadb
user_table = users
charset = utf8
db_category = category
db_family = family
db_platform = platform
db_benign = benign
db_domain = domain
db_component = api_component
```

5. **配置AI服务**(可选)

如需使用AI分析功能,在 `config.ini` 中配置:
```ini
[flowviz]
openai_api_key = your_openai_api_key
openai_base_url = https://api.openai.com/v1
openai_model = gpt-4o
claude_api_key = your_claude_api_key
claude_base_url = https://api.anthropic.com
claude_model = claude-3-opus-20240229
default_ai_provider = openai
```

6. **启动服务**

开发模式:
```bash
python main.py
```

生产模式:
```bash
uvicorn main:app --host 0.0.0.0 --port 5005 --workers 4
```

7. **访问应用**
- API文档: http://127.0.0.1:5005/docs
- 健康检查: http://127.0.0.1:5005/health
- 根路径: http://127.0.0.1:5005/

## API文档

### 认证API

```http
POST /api/login              # 用户登录
POST /api/register           # 用户注册
GET /api/user/info           # 获取用户信息
```

### ATT&CK矩阵API

```http
GET /api/attck/matrix                    # 获取ATT&CK矩阵
GET /api/attck/technique-mapping         # 获取技术映射
GET /api/attck/api-components            # 获取API组件
GET /api/attck/technique/{technique_id}  # 获取技术详情
GET /api/attck/function/{function_name}  # 获取函数详情
```

### FlowViz API

```http
POST /flowviz/api/analyze-stream  # 流式分析
POST /flowviz/api/export          # 导出结果
GET /flowviz/api/providers        # 获取AI提供商列表
GET /flowviz/api/history          # 获取历史记录
```

### 检测API

```http
POST /api/detect/file            # 文件检测
POST /api/detect/hash            # 哈希检测
GET /api/detect/result/{task_id} # 获取检测结果
POST /api/detect/vt              # VirusTotal检测
```

### 查询API

```http
GET /api/query/sample/{sha256}   # 查询样本信息
GET /api/query/download/{sha256} # 下载样本文件
GET /api/query/domain/{domain}   # 查询域名信息
```

### 配置API

```http
GET /api/config                  # 获取所有配置
PUT /api/config                  # 更新配置
GET /api/config/{key}            # 获取单个配置
```

## FlowViz功能

### 流式分析

FlowViz支持实时流式分析,自动提取攻击流程:

- **支持输入**: 文本报告、威胁情报、结构化JSON数据
- **AI分析**: OpenAI GPT-4o, Anthropic Claude
- **节点类型**: action, tool, malware, asset, infrastructure, url, vulnerability
- **边类型**: Uses, Targets, Communicates with, Leads to

### 导出格式

- **JSON**: 标准JSON格式
- **PNG**: 流程图图片
- **ATT&CK Flow v3**: MITRE ATT&CK Flow格式
- **STIX**: 威胁情报标准格式

### 高级特性

- **严格验证**: 确保输出符合ATT&CK规范
- **技术映射**: 自动映射到ATT&CK技术ID
- **视觉分析**: 支持图像输入分析
- **历史记录**: 保存分析历史,支持回顾

## 定时任务

系统内置定时任务,自动生成前端数据:

- **首次启动**: 立即执行一次数据生成
- **每日任务**: 凌晨2点自动更新
- **测试任务**: 每6小时更新一次

生成的数据包括:
- 恶意样本统计(总数、年度、月度)
- 良性样本统计
- 域名统计(总数、近一年、近30天)
- 分类统计(Category、Platform、Family)

## DGA检测

系统支持多种机器学习算法进行DGA检测:

- **KNN**: K近邻分类器
- **朴素贝叶斯**: 朴素贝叶斯分类器
- **逻辑回归**: 逻辑回归分类器
- **决策树**: 决策树分类器
- **随机森林**: 随机森林分类器
- **AdaBoost**: AdaBoost分类器
- **GBDT**: 梯度提升决策树
- **XGBoost**: 极端梯度提升
- **SVM**: 支持向量机
- **LSTM**: 长短期记忆网络

## 配置说明

### 数据库配置

```ini
[mysql]
host = 127.0.0.1
port = 3306
user = your_mysql_user
passwd = your_mysql_password
db = malicious
db_web = webdatadb
user_table = users
charset = utf8
db_category = category
db_family = family
db_platform = platform
db_benign = benign
db_domain = domain
db_component = api_component
```

### 应用配置

```ini
[ini]
ip = 0.0.0.0
port = 5005
row_per_page = 20
```

### JWT配置

```ini
[jwt]
secret = CHANGE_THIS_TO_A_SECURE_RANDOM_STRING
expire_hours = 24
```

### 文件上传配置

```ini
[flowviz]
max_request_size = 10485760      # 10MB
max_image_size = 3145728         # 3MB
max_article_size = 5242880       # 5MB
```

### Docker模型配置

```ini
[docker_models]
api_base = http://127.0.0.1
ember = 8000
malconv = 8001
imcfn = 8002
malconv2 = 8003
transformer = 8004
inceptionv3 = 8005
rcnf = 8006
onedense_cnn = 8007
malgraph = 8008
timeout = 30
```

## 开发指南

### 添加新API

1. 在`app/api/`创建新的路由文件
2. 在`app/api/__init__.py`中注册路由
3. 在`app/schemas/`定义数据模型
4. 在`app/services/`实现业务逻辑

```python
# app/api/my_api.py
from fastapi import APIRouter, Depends
from app.schemas.my_schema import MyResponse

router = APIRouter()

@router.get("/my-endpoint", response_model=MyResponse)
async def my_endpoint():
    # 实现业务逻辑
    pass
```

### 添加新的AI提供商

1. 在`app/services/flowviz/providers/`创建新的提供商文件
2. 继承`BaseProvider`类
3. 实现`stream()`方法
4. 在`app/services/flowviz/providers/factory.py`注册提供商

```python
# app/services/flowviz/providers/my_provider.py
from .base import BaseProvider

class MyProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__(api_key, base_url, model)

    async def stream(self, prompt: str):
        # 实现流式分析
        pass
```

### 添加新的机器学习分类器

1. 在`feeds/`目录创建新的分类器文件
2. 实现训练和预测接口
3. 在`config.ini`的`[feeds]`部分注册

## 性能优化

- **异步处理**: 使用FastAPI的异步特性提升性能
- **连接池**: 数据库连接池优化
- **流式响应**: SSE流式传输减少内存占用
- **定时任务**: 后台自动更新数据,减少实时查询压力
- **缓存机制**: 减少重复计算

## 安全特性

- **JWT认证**: 无状态令牌认证
- **密码加密**: bcrypt密码哈希
- **SQL注入防护**: 参数化查询
- **XSS防护**: 输入验证和转义
- **CORS配置**: 跨域访问控制
- **文件上传验证**: 文件类型和大小限制

## 测试

```bash
# 运行所有API测试
python test_all_api.py

# 测试特定模块
pytest tests/test_auth.py
pytest tests/test_flowviz.py
```

## 日志

日志文件位于`logs/`目录:
- `app.log`: 应用日志
- `flowviz.log`: FlowViz模块日志

日志级别可在`config.ini`中配置:
```ini
log_level = INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## 常见问题

### 1. 数据库连接失败
检查MySQL服务是否启动,配置是否正确:
```bash
# 检查MySQL服务状态
systemctl status mysql  # Linux
# 或
net start mysql  # Windows
```

### 2. AI分析失败
检查API密钥是否有效,网络是否通畅:
- 确认API密钥正确
- 检查网络连接
- 查看日志文件获取详细错误信息

### 3. FlowViz解析失败
检查AI响应格式,查看日志文件:
- 确认AI提供商配置正确
- 检查输入数据格式
- 查看日志获取详细错误信息

### 4. 前端数据不更新
检查定时任务是否正常运行,查看日志输出:
```bash
# 查看日志
tail -f logs/app.log
```

### 5. 文件上传失败
检查文件大小和类型限制:
- 确认文件大小在限制范围内
- 检查文件类型是否允许
- 查看磁盘空间是否充足

### 6. 模型加载失败
检查模型文件是否存在,依赖是否完整:
- 确认模型文件在正确路径
- 检查机器学习库是否正确安装
- 查看日志获取详细错误信息

## 更新日志

### v2.0.0 (2026-03-11)
- 从Flask迁移到FastAPI
- 实现FlowViz流式分析
- 优化ATT&CK矩阵查询
- 改进用户认证系统
- 添加结构化数据处理
- 添加定时任务自动更新前端数据
- 清理冗余代码和文件
- 完善文档和配置示例

### v1.0.0 (2025-02-28)
- 初始版本
- Flask后端实现
- 基础功能开发

## 许可证

本项目采用 MIT 许可证