# 恶意样本分析系统 - Vue3版本

## 项目简介

这是一个基于Vue3 + Vite + Element Plus的恶意样本分析系统前端项目,提供样本检测、ATT&CK矩阵分析、攻击流可视化等功能。

## 技术栈

- **Vue 3.4.0** - 渐进式JavaScript框架
- **Vite 5.0.11** - 下一代前端构建工具
- **Vue Router 4.2.5** - Vue.js官方路由
- **Vuex 4.1.0** - Vue.js状态管理
- **Element Plus** - 基于Vue 3的组件库
- **ECharts 5** - 数据可视化图表库
- **Axios** - HTTP请求库

## 主要功能

### 1. 样本检测
- 样本上传与分析
- 检测结果展示
- 历史记录查询

### 2. ATT&CK矩阵分析
- 企业矩阵视图
- 技术详情查看
- API组件映射
- 技术映射管理
- 函数详情分析(含C++代码高亮)

### 3. 攻击流可视化
- 流程图可视化
- 节点详情分析
- 历史记录管理
- 导出功能

### 4. 数据可视化
- 样本统计图表
- 域名分析
- 攻击趋势分析

## 项目结构

```
vue/
├── public/              # 静态资源
├── src/
│   ├── api/            # API接口
│   ├── assets/         # 资源文件
│   ├── components/     # 公共组件
│   ├── icons/          # SVG图标
│   ├── layout/         # 布局组件
│   ├── router/         # 路由配置
│   ├── store/          # Vuex状态管理
│   ├── styles/         # 全局样式
│   ├── utils/          # 工具函数
│   └── views/          # 页面组件
│       ├── attck/      # ATT&CK相关页面
│       ├── dashboard/  # 仪表盘
│       ├── flowviz/    # 攻击流可视化
│       └── ...
├── .env.development    # 开发环境变量
├── .env.production     # 生产环境变量
├── vite.config.js      # Vite配置
└── package.json        # 项目依赖
```

## 快速开始

### 安装依赖

```bash
npm install
```

### 开发环境运行

```bash
npm run dev
```

访问: http://localhost:9528

### 生产环境构建

```bash
npm run build
```

### 代码检查

```bash
npm run lint
```

## 环境配置

项目使用`.env`文件管理环境变量:

**.env.development** (开发环境)
```bash
ENV = 'development'
VITE_APP_BASE_API = 'http://YOUR_SERVER_IP:5005'
```

**.env.production** (生产环境)
```bash
ENV = 'production'
VITE_APP_BASE_API = 'http://YOUR_SERVER_IP:5005'
```

后端启动时会自动更新这些文件中的API地址。

## API代理配置

开发环境下,Vite会自动代理API请求:

```javascript
// vite.config.js
proxy: {
  '/dev-api': {
    target: 'http://YOUR_SERVER_IP:5005',
    changeOrigin: true,
  },
  '/api': {
    target: 'http://YOUR_SERVER_IP:5005',
    changeOrigin: true,
  },
  '/flowviz': {
    target: 'http://YOUR_SERVER_IP:5005',
    changeOrigin: true,
  }
}
```

**注意**: 实际的API地址由`.env`文件中的`VITE_APP_BASE_API`变量控制。

## Vue2到Vue3迁移说明

本项目已从Vue2成功迁移到Vue3,主要变更:

### 1. 构建工具
- Vue CLI → Vite
- Webpack → Vite内置打包器

### 2. 核心依赖
- Vue 2.6.10 → Vue 3.4.0
- Element UI → Element Plus
- Vue Router 3 → Vue Router 4
- Vuex 3 → Vuex 4

### 3. 语法变更
- `slot-scope="scope"` → `#default="scope"`
- `:visible.sync="dialogVisible"` → `v-model="dialogVisible"`
- 过滤器已移除,改用方法调用
- `$listeners`已移除,合并到`$attrs`

### 4. 配置文件
- `vue.config.js` → `vite.config.js`
- `config.ini` → `.env`文件

详细的迁移说明请参考Vue3官方文档。

## 后端接口

后端使用Flask提供API服务,主要接口:

- `/api/login` - 用户登录
- `/api/detect` - 样本检测
- `/api/search` - 样本搜索
- `/dev-api/api/attck/*` - ATT&CK相关接口
- `/flowviz/api` - 攻击流可视化接口

## 浏览器支持

- Chrome (推荐)
- Firefox
- Safari
- Edge

不支持IE浏览器。

## 开发建议

1. 使用VSCode + Volar插件
2. 遵循Vue3官方风格指南
3. 组件命名使用PascalCase
4. 使用Composition API (推荐) 或 Options API

## 常见问题

### 1. 依赖安装失败
```bash
# 清除缓存重新安装
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### 2. 开发服务器启动失败
检查端口9528是否被占用,或修改`vite.config.js`中的端口号。

### 3. API请求失败
- 检查后端服务是否启动
- 检查`.env`文件中的API地址是否正确
- 检查浏览器控制台的网络请求

## 许可证

MIT License

## 联系方式

如有问题,请提交Issue或联系开发团队。
