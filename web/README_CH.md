# 恶意样本分析系统

[English](./README_EN.md) | 中文

基于 FastAPI + Vue3 构建的全栈恶意样本分析系统，提供恶意软件检测、ATT&CK 矩阵分析、FlowViz 攻击流程可视化等功能。

## 目录

- [项目概述](#项目概述)
- [主要功能](#主要功能)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心功能详解](#核心功能详解)
- [API 文档](#api-文档)
- [配置说明](#配置说明)
- [开发指南](#开发指南)
- [部署指南](#部署指南)
- [性能优化](#性能优化)
- [安全特性](#安全特性)
- [测试](#测试)
- [常见问题](#常见问题)
- [更新日志](#更新日志)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 项目概述

本系统是一个完整的恶意样本分析平台，采用前后端分离架构，集成了多种先进的机器学习模型和 AI 分析能力。

### 核心特性

- **前后端分离**: FastAPI 后端 + Vue3 前端
- **Docker 容器化**: 9 种机器学习模型容器化部署
- **AI 驱动**: 集成 OpenAI GPT-4o 和 Anthropic Claude
- **实时分析**: 支持流式 AI 分析和实时数据更新
- **可视化**: ATT&CK 矩阵可视化和攻击流程图

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端层 (Vue3)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 样本检测  │  │ATT&CK矩阵│  │ FlowViz  │  │ 数据统计  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      后端层 (FastAPI)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 认证服务  │  │ 检测服务  │  │ AI服务   │  │ 数据服务  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        ┌───────────┐  ┌───────────┐  ┌───────────┐
        │  MySQL    │  │  Docker   │  │  AI API   │
        │  数据库    │  │  模型服务  │  │  服务     │
        └───────────┘  └───────────┘  └───────────┘
```

## 主要功能

### 1. 恶意软件检测

- **多引擎集成检测**: 9 种机器学习模型并行检测
  - Ember (端口 8000) - 基于特征的恶意软件检测
  - Malconv (端口 8001) - 基于卷积神经网络的检测
  - Imcfn (端口 8002) - 改进的多尺度卷积融合网络
  - Malconv2 (端口 8003) - Malconv 的改进版本
  - Transformer (端口 8004) - 基于 Transformer 架构的检测模型
  - InceptionV3 (端口 8005) - 基于 InceptionV3 的恶意软件检测
  - Rcnf (端口 8006) - 残差卷积网络融合模型
  - 1D-CNN (端口 8007) - 一维卷积神经网络
  - Malgraph (端口 8008) - 基于图神经网络的恶意软件检测

- **Docker 容器化部署**: 所有模型通过 Docker API 提供服务
- **集成预测机制**: 多模型投票、概率集成、容错机制
- **支持多种输入**: 文件上传、哈希查询

### 2. ATT&CK 矩阵分析

- **企业矩阵视图**: 完整的 MITRE ATT&CK 框架可视化
- **技术详情查看**: 详细的技术描述和检测方法
- **API 组件映射**: 技术与 API 组件的映射关系
- **技术映射管理**: 支持自定义技术映射
- **函数详情分析**: 含 C++ 代码高亮显示

### 3. FlowViz 攻击流程可视化

- **实时流式分析**: 支持 SSE 流式传输
- **AI 驱动分析**: OpenAI GPT-4o 和 Anthropic Claude
- **多种输入格式**: 文本报告、威胁情报、结构化 JSON
- **丰富的节点类型**: action, tool, malware, asset, infrastructure, url, vulnerability
- **多种导出格式**: JSON, PNG, ATT&CK Flow v3, STIX

### 4. 数据可视化

- **样本统计图表**: 恶意样本和良性样本统计
- **域名分析**: 恶意域名库管理和 DGA 检测
- **攻击趋势分析**: 年度、月度趋势分析
- **分类统计**: Category、Platform、Family 分类统计

### 5. 用户管理

- **JWT 认证**: 无状态令牌认证机制
- **权限管理**: 基于角色的访问控制
- **用户注册登录**: 完整的用户认证流程

## 技术架构

### 后端技术栈

| 技术 | 版本 | 说明 |
|------|------|------|
| FastAPI | 0.115.0 | 现代高性能 Web 框架 |
| MySQL | 8.0 | 关系型数据库 |
| PyJWT | 2.10.0 | JWT 令牌认证 |
| APScheduler | 3.10.0+ | 定时任务调度 |
| aiohttp | 3.11.0 | 异步 HTTP 客户端 |
| Docker | - | 容器化模型服务 |

### 前端技术栈

| 技术 | 版本 | 说明 |
|------|------|------|
| Vue | 3.4.0 | 渐进式 JavaScript 框架 |
| Vite | 5.0.11 | 下一代前端构建工具 |
| Element Plus | 2.5.0 | Vue 3 组件库 |
| ECharts | 5.4.3 | 数据可视化图表库 |
| Vuex | 4.1.0 | 状态管理 |
| Vue Router | 4.2.5 | 路由管理 |

### AI 服务

- **OpenAI GPT-4o**: FlowViz 攻击流程分析
- **Anthropic Claude**: 备用 AI 分析引擎

## 项目结构

```
web/
├── fastapi/                      # FastAPI 后端
│   ├── app/                      # 应用主目录
│   │   ├── api/                  # API 路由
│   │   │   ├── auth.py          # 用户认证 API
│   │   │   ├── attck.py         # ATT&CK 矩阵 API
│   │   │   ├── detect.py        # 恶意软件检测 API
│   │   │   ├── flowviz.py       # FlowViz 基础 API
│   │   │   ├── flowviz_streaming.py  # FlowViz 流式分析 API
│   │   │   ├── query.py         # 查询 API
│   │   │   └── config.py        # 配置管理 API
│   │   ├── core/                 # 核心配置
│   │   │   ├── config.py        # 应用配置
│   │   │   └── database.py      # 数据库连接
│   │   ├── schemas/              # 数据模型
│   │   ├── scripts/              # 业务脚本
│   │   │   ├── config_manager.py    # 配置管理
│   │   │   ├── dga_detection.py     # DGA 检测
│   │   │   ├── file_detect.py       # 文件检测
│   │   │   └── train_model.py       # 模型训练
│   │   ├── services/             # 业务服务
│   │   │   ├── auth_service.py      # 认证服务
│   │   │   ├── flowviz_service.py   # FlowViz 服务
│   │   │   ├── vue_data_service.py  # 数据服务
│   │   │   ├── detection/           # 检测服务
│   │   │   │   └── ensemble_predict.py  # Docker 模型集成预测
│   │   │   └── flowviz/             # FlowViz 模块
│   │   └── utils/                # 工具函数
│   ├── data/                     # 数据文件
│   │   ├── model/               # 训练好的模型
│   │   ├── features/            # 特征文件
│   │   └── sample/              # 样本数据
│   ├── feeds/                    # 机器学习分类器
│   │   ├── svm.py               # SVM 分类器
│   │   ├── knn.py               # KNN 分类器
│   │   ├── lstm.py              # LSTM 分类器
│   │   └── ...                  # 其他分类器
│   ├── attack/                   # ATT&CK 数据
│   │   ├── enterprise-attack.json  # ATT&CK 数据
│   │   └── download_parse_import.py # 数据导入脚本
│   ├── uploads/                  # 上传文件
│   ├── logs/                     # 日志文件
│   ├── main.py                   # 应用入口
│   ├── config.ini                # 配置文件
│   └── requirements.txt          # Python 依赖
│
├── vue/                          # Vue3 前端
│   ├── public/                   # 静态资源
│   ├── src/
│   │   ├── api/                  # API 接口
│   │   ├── components/           # 公共组件
│   │   ├── layout/               # 布局组件
│   │   ├── router/               # 路由配置
│   │   ├── store/                # Vuex 状态管理
│   │   ├── styles/               # 全局样式
│   │   ├── utils/                # 工具函数
│   │   └── views/                # 页面组件
│   │       ├── attck/           # ATT&CK 相关页面
│   │       ├── dashboard/       # 仪表盘
│   │       ├── flowviz/         # 攻击流可视化
│   │       └── ...
│   ├── .env.development          # 开发环境变量
│   ├── .env.production           # 生产环境变量
│   ├── vite.config.js            # Vite 配置
│   └── package.json              # 前端依赖
│
├── README.md                     # 中文文档
├── README_EN.md                  # 英文文档
└── .gitignore                    # Git 忽略文件
```

## 快速开始

### 环境要求

- **Python**: 3.9+
- **Node.js**: 16+
- **MySQL**: 8.0+
- **Docker**: 最新版本（用于机器学习模型服务）

### 后端部署

#### 1. 克隆项目

```bash
git clone <repository_url>
cd web/fastapi
```

#### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置数据库和 Docker 服务

编辑 `config.ini` 文件：

```ini
[mysql]
host = localhost
port = 3306
user = root
passwd = your_password
db = malware_analysis
db_component = api_component
db_benign = benign_samples
db_domain = db_domain

[docker_models]
# Docker API 基础地址
api_base = http://192.168.8.202

# Docker 模型端口配置
ember = 8000
malconv = 8001
imcfn = 8002
malconv2 = 8003
transformer = 8004
inceptionv3 = 8005
rcnf = 8006
onedense_cnn = 8007
malgraph = 8008

# API 超时时间（秒）
timeout = 30

[flowviz]
# AI 服务配置
openai_api_key = your_openai_key
openai_base_url = https://api.openai.com/v1
openai_model = gpt-4o
claude_api_key = your_claude_key
claude_model = claude-sonnet-4-5-20250929
default_ai_provider = openai
```

#### 5. 启动后端服务

```bash
# 开发模式
python main.py

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 5005 --workers 4
```

#### 6. 访问 API 文档

- Swagger UI: http://localhost:5005/docs
- ReDoc: http://localhost:5005/redoc
- 健康检查: http://localhost:5005/health

### 前端部署

#### 1. 进入前端目录

```bash
cd web/vue
```

#### 2. 安装依赖

```bash
npm install
```

#### 3. 配置环境变量

编辑 `.env.development` 或 `.env.production`：

```bash
ENV = 'development'
VITE_APP_BASE_API = 'http://localhost:5005'
```

#### 4. 启动开发服务器

```bash
npm run dev
```

访问: http://localhost:9528

#### 5. 生产环境构建

```bash
npm run build
```

构建产物将生成在 `dist/` 目录。

## 核心功能详解

### Docker 模型服务

系统采用 Docker 容器化部署机器学习模型，提供更灵活、可扩展的检测服务。

#### 架构优势

- **资源隔离**: 每个模型独立运行，互不干扰
- **易于扩展**: 可轻松添加新的检测模型
- **负载均衡**: 可根据需求调整各模型的资源分配
- **版本管理**: 支持模型版本更新和回滚
- **高可用性**: 单个模型故障不影响整体服务

#### 集成预测机制

```python
# 多模型投票机制
if malicious_count > safe_count:
    ensemble_result = '恶意'
    ensemble_prob = max(malicious_probs)
else:
    ensemble_result = '安全'
    ensemble_prob = min(safe_probs)
```

#### 添加新模型

1. 构建新的 Docker 镜像并部署
2. 在 `config.ini` 添加端口配置
3. 在 `ensemble_predict.py` 注册模型
4. 重启 FastAPI 服务

### FlowViz 攻击流程分析

FlowViz 是本系统的核心功能之一，支持实时流式分析攻击流程。

#### 工作流程

```
输入数据 → AI 分析 → 节点提取 → 关系构建 → 图可视化 → 导出
```

#### 支持的输入格式

- **文本报告**: 威胁情报报告、安全事件描述
- **结构化 JSON**: STIX 格式、自定义 JSON 格式
- **混合数据**: 文本 + JSON 混合输入

#### 节点类型

| 类型 | 说明 | 示例 |
|------|------|------|
| action | 攻击行为 | "执行恶意代码" |
| tool | 攻击工具 | "Mimikatz" |
| malware | 恶意软件 | "Emotet" |
| asset | 目标资产 | "域控制器" |
| infrastructure | 基础设施 | "C2服务器" |
| url | 恶意链接 | "http://malicious.com" |
| vulnerability | 漏洞 | "CVE-2021-44228" |

#### 导出格式

- **JSON**: 标准图数据格式
- **PNG**: 高清流程图
- **ATT&CK Flow v3**: MITRE ATT&CK Flow 格式
- **STIX**: 威胁情报标准格式

### 定时任务

系统内置定时任务，自动生成前端数据：

- **首次启动**: 立即执行一次数据生成
- **每日任务**: 凌晨 2 点自动更新
- **测试任务**: 每 6 小时更新一次

生成的数据包括：

- 恶意样本统计（总数、年度、月度）
- 良性样本统计
- 域名统计（总数、近一年、近 30 天）
- 分类统计（Category、Platform、Family）

## API 文档

### 认证 API

```http
POST /api/login              # 用户登录
POST /api/register           # 用户注册
GET /api/user/info           # 获取用户信息
PUT /api/user/info           # 更新用户信息
```

### 检测 API

```http
POST /api/detect/file            # 文件检测（9 种 Docker 模型集成预测）
POST /api/detect/hash            # 哈希检测
GET /api/detect/result/{task_id} # 获取检测结果
GET /api/detect/history          # 获取检测历史
```

**检测模型**: 通过 Docker 容器化部署的 9 种机器学习模型

### ATT&CK 矩阵 API

```http
GET /api/attck/matrix                    # 获取 ATT&CK 矩阵
GET /api/attck/technique-mapping         # 获取技术映射
GET /api/attck/api-components            # 获取 API 组件
GET /api/attck/technique/{technique_id}  # 获取技术详情
POST /api/attck/technique-mapping        # 创建技术映射
PUT /api/attck/technique-mapping/{id}    # 更新技术映射
```

### FlowViz API

```http
POST /flowviz/api/analyze-stream  # 流式分析（SSE）
POST /flowviz/api/analyze         # 同步分析
POST /flowviz/api/export          # 导出结果
GET /flowviz/api/providers        # 获取 AI 提供商列表
GET /flowviz/api/history          # 获取分析历史
```

### 查询 API

```http
GET /api/query/sample/{sha256}   # 查询样本信息
GET /api/query/download/{sha256} # 下载样本文件
GET /api/query/search            # 搜索样本
```

### 配置 API

```http
GET /api/config                  # 获取所有配置
PUT /api/config                  # 更新配置
GET /api/config/{key}            # 获取单个配置
```

## 配置说明

### 数据库配置

```ini
[mysql]
host = localhost
port = 3306
user = root
passwd = your_password
db = malware_analysis
db_component = api_component
db_benign = benign_samples
db_domain = db_domain
charset = utf8
```

### Docker 模型配置

```ini
[docker_models]
api_base = http://127.0.1.1
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

### AI 服务配置

```ini
[flowviz]
openai_api_key = your_openai_key
openai_base_url = https://api.openai.com/v1
openai_model = gpt-4o
claude_api_key = your_claude_key
claude_base_url = https://api.anthropic.com
claude_model = claude-sonnet-4-5-20250929
default_ai_provider = openai
```

### 应用配置

```ini
[server]
host = 0.0.0.0
port = 5005
debug = True

[app]
name = 恶意样本分析系统
version = 2.0.0

[jwt]
secret = your_jwt_secret_key
expire_hours = 24
```

## 开发指南

### 后端开发

#### 添加新 API

1. 在 `app/api/` 创建新的路由文件
2. 在 `app/api/__init__.py` 中注册路由
3. 在 `app/schemas/` 定义数据模型
4. 在 `app/services/` 实现业务逻辑

#### 添加新的 AI 提供商

1. 在 `app/services/flowviz/providers/` 创建新的提供商文件
2. 继承 `BaseProvider` 类
3. 实现 `stream()` 方法
4. 在 `app/services/flowviz/providers/factory.py` 注册提供商

#### 添加新的 Docker 检测模型

1. 构建新的 Docker 镜像并部署到服务器
2. 在 `config.ini` 的 `[docker_models]` 部分添加模型端口配置
3. 在 `app/services/detection/ensemble_predict.py` 的 `DOCKER_MODELS` 字典中添加模型信息
4. 重启 FastAPI 服务即可

### 前端开发

#### 添加新页面

1. 在 `src/views/` 创建页面组件
2. 在 `src/api/` 添加 API 接口
3. 在 `src/router/` 配置路由
4. 在 `src/store/` 管理状态

#### 代码规范

- 使用 ESLint 进行代码检查
- 遵循 Vue 3 官方风格指南
- 组件命名使用 PascalCase
- 优先使用 Composition API

## 部署指南

### 开发环境启动

#### 后端启动

```bash
# 进入后端目录
cd fastapi

# 激活虚拟环境（如果使用）
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 启动后端服务
python main.py
```

后端服务将在 `http://0.0.0.0:5005` 启动，并自动：
- 创建必要的目录
- 更新 Vue 前端配置文件
- 启动定时任务（首次运行会立即生成前端数据）
- 启动 API 服务

#### 前端启动

```bash
# 进入前端目录
cd vue

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

前端开发服务器将在 `http://localhost:9528` 启动。

**注意**: 后端启动时会自动更新前端的 `.env.development` 和 `.env.production` 文件，配置正确的 API 地址。

### 生产环境部署

#### 后端部署

```bash
# 进入后端目录
cd fastapi

# 使用 Uvicorn 生产模式启动
uvicorn main:app --host 0.0.0.0 --port 5005 --workers 4
```

**生产环境建议**:
- 使用进程管理工具（如 systemd、supervisor）管理后端服务
- 配置日志轮转
- 设置环境变量 `DEBUG=False`

#### 前端部署

```bash
# 进入前端目录
cd vue

# 构建生产版本
npm run build

# 构建产物在 dist/ 目录
# 将 dist/ 目录部署到 Web 服务器
```

**Web 服务器配置示例** (Nginx):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /path/to/dist;
        try_files $uri $uri/ /index.html;
    }

    # API 代理
    location /api {
        proxy_pass http://localhost:5005;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # FlowViz API 代理
    location /flowviz {
        proxy_pass http://localhost:5005;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Docker 模型服务部署

机器学习模型需要单独部署为 Docker 容器服务。每个模型运行在独立的容器中，通过 HTTP API 提供预测服务。

#### 模型容器部署

确保以下 Docker 容器正在运行：

```bash
# Ember 模型 (端口 8000)
docker run -d -p 8000:8000 --name ember-model ember-image

# Malconv 模型 (端口 8001)
docker run -d -p 8001:8001 --name malconv-model malconv-image

# Imcfn 模型 (端口 8002)
docker run -d -p 8002:8002 --name imcfn-model imcfn-image

# Malconv2 模型 (端口 8003)
docker run -d -p 8003:8003 --name malconv2-model malconv2-image

# Transformer 模型 (端口 8004)
docker run -d -p 8004:8004 --name transformer-model transformer-image

# InceptionV3 模型 (端口 8005)
docker run -d -p 8005:8005 --name inceptionv3-model inceptionv3-image

# Rcnf 模型 (端口 8006)
docker run -d -p 8006:8006 --name rcnf-model rcnf-image

# 1D-CNN 模型 (端口 8007)
docker run -d -p 8007:8007 --name onedense-cnn-model onedense-cnn-image

# Malgraph 模型 (端口 8008)
docker run -d -p 8008:8008 --name malgraph-model malgraph-image
```

**注意**: 具体的 Docker 镜像名称和启动参数需要根据实际的模型镜像配置。

### 系统服务配置（Linux）

#### 创建 systemd 服务

创建后端服务文件 `/etc/systemd/system/malware-backend.service`:

```ini
[Unit]
Description=Malware Analysis Backend
After=network.target mysql.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/web/fastapi
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn main:app --host 0.0.0.0 --port 5005
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable malware-backend
sudo systemctl start malware-backend
sudo systemctl status malware-backend
```

## 性能优化

### 后端优化

- **异步处理**: 使用 FastAPI 的异步特性提升性能
- **连接池**: 数据库连接池优化
- **流式响应**: SSE 流式传输减少内存占用
- **定时任务**: 后台自动更新数据，减少实时查询压力
- **缓存机制**: Redis 缓存热点数据

### 前端优化

- **代码分割**: 路由懒加载
- **资源压缩**: Gzip 压缩
- **CDN 加速**: 静态资源 CDN 部署
- **图片优化**: WebP 格式、懒加载

### 数据库优化

- **索引优化**: 为常用查询字段添加索引
- **查询优化**: 避免 N+1 查询
- **分库分表**: 大表分片存储

## 安全特性

### 认证与授权

- **JWT 认证**: 无状态令牌认证
- **密码加密**: bcrypt 密码哈希
- **权限控制**: 基于角色的访问控制

### 数据安全

- **SQL 注入防护**: 参数化查询
- **XSS 防护**: 输入验证和转义
- **CSRF 防护**: Token 验证
- **文件上传安全**: 文件类型验证、大小限制

### 网络安全

- **HTTPS**: 强制 HTTPS 加密传输
- **CORS 配置**: 跨域访问控制
- **Rate Limiting**: API 请求频率限制

## 测试

### 后端测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_auth.py
pytest tests/test_flowviz.py

# 生成覆盖率报告
pytest --cov=app tests/
```

### 前端测试

```bash
# 运行单元测试
npm run test:unit

# 运行端到端测试
npm run test:e2e

# 生成覆盖率报告
npm run test:coverage
```

## 常见问题

### 1. 数据库连接失败

**问题**: 启动时报数据库连接错误

**解决方案**:
- 检查 MySQL 服务是否启动
- 检查 `config.ini` 中的数据库配置是否正确
- 检查数据库用户权限

### 2. Docker 模型服务连接失败

**问题**: 检测功能无法使用

**解决方案**:
- 检查 Docker 服务是否正常运行
- 检查 `config.ini` 中的 `docker_models` 配置是否正确
- 检查各模型容器的端口是否可访问
- 查看 Docker 容器日志排查问题

### 3. AI 分析失败

**问题**: FlowViz 分析失败

**解决方案**:
- 检查 API 密钥是否有效
- 检查网络是否通畅
- 检查 AI 服务配额是否充足

### 4. 前端数据不更新

**问题**: 前端统计数据不更新

**解决方案**:
- 检查定时任务是否正常运行
- 查看后端日志输出
- 手动触发数据更新

### 5. 文件上传失败

**问题**: 上传文件时报错

**解决方案**:
- 检查文件大小是否超过限制
- 检查文件类型是否支持
- 检查 `uploads/` 目录权限

## 更新日志

### v2.0.0 (2026-03-08)

**重大更新**:
- 后端从 Flask 迁移到 FastAPI
- 前端从 Vue2 迁移到 Vue3
- 机器学习模型 Docker 化部署

**新功能**:
- 实现 FlowViz 流式分析
- 优化 ATT&CK 矩阵查询
- 改进用户认证系统
- 添加结构化数据处理
- 添加定时任务自动更新前端数据

**优化**:
- 移除本地模型依赖，删除 models 目录
- 清理冗余代码和文件
- 性能优化和代码重构

### v1.0.0 (2025-02-28)

- 初始版本
- Flask + Vue2 实现
- 基础功能开发


### 代码规范

- 遵循 PEP 8（Python）和 ESLint（JavaScript）代码规范
- 编写清晰的提交信息
- 添加必要的测试用例
- 更新相关文档

### 问题反馈

如果您发现 Bug 或有功能建议，请提交 Issue。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- **项目维护者**: NKAMG
- **项目地址**: 内部项目
- **技术支持**: 提交 Issue 或联系开发团队

---

**详细文档**:
- [后端文档](./fastapi/README.md)
- [前端文档](./vue/README.md)
