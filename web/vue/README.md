# 恶意样本分析系统 - Vue3前端

基于Vue3 + Vite + Element Plus的恶意样本分析系统前端项目,提供样本检测、分布式杀毒扫描、ATT&CK矩阵分析、攻击流可视化等功能。

## 项目简介

这是一个现代化的恶意样本分析系统前端,采用Vue3生态系统构建,提供直观的用户界面和丰富的交互体验。

### 主要特性

- **现代化架构**: 基于Vue3 Composition API和Vite构建工具
- **丰富的组件库**: 基于Element Plus的UI组件
- **数据可视化**: 集成ECharts提供丰富的图表展示
- **流程图可视化**: 支持Dagre和D3.js的流程图渲染
- **分布式杀毒扫描**: 支持多虚拟机分布式杀毒扫描,实时进度显示
- **响应式设计**: 适配不同屏幕尺寸
- **代码高亮**: 支持多种编程语言的代码高亮
- **文件处理**: 支持文件上传、下载和预览
- **国际化**: 支持中英文切换

## 技术栈

### 核心框架
- **Vue 3.4.0**: 渐进式JavaScript框架
- **Vite 5.0.11**: 下一代前端构建工具
- **Vue Router 4.2.5**: Vue.js官方路由
- **Vuex 4.1.0**: Vue.js状态管理

### UI组件库
- **Element Plus 2.5.0**: 基于Vue 3的组件库
- **normalize.css 7.0.0**: CSS重置

### 数据可视化
- **ECharts 5.4.3**: 数据可视化图表库
- **ECharts GL 2.0.9**: 3D图表扩展
- **ECharts Map 3.0.1**: 地图图表扩展
- **Dagre 0.8.5**: 有向图布局算法
- **Dagre D3 Renderer 0.5.8**: Dagre的D3渲染器
- **Graphlib 2.1.8**: 图论库

### 工具库
- **Axios 1.11.0**: HTTP请求库
- **js-cookie 3.0.5**: Cookie管理
- **file-saver 2.0.1**: 文件保存
- **jszip 3.7.0**: ZIP文件处理
- **xlsx 0.17.0**: Excel文件处理
- **html2canvas 1.4.1**: 截图功能
- **nprogress 0.2.0**: 进度条
- **screenfull 6.0.2**: 全屏功能
- **clipboard 2.0.4**: 剪贴板操作
- **decimal.js 10.6.0**: 精确数值计算
- **dompurify 3.3.1**: XSS防护

### 编辑器组件
- **CodeMirror 5.58.2**: 代码编辑器
- **Tinymce**: 富文本编辑器
- **Markdown Editor**: Markdown编辑器
- **Json Editor**: JSON编辑器

### 拖拽与排序
- **vuedraggable 4.1.0**: 拖拽组件
- **sortablejs 1.15.0**: 排序库
- **dropzone 5.5.1**: 文件拖拽上传

### 其他组件
- **driver.js 0.9.5**: 页面引导
- **fuse.js 3.4.4**: 模糊搜索
- **vue-splitpane 1.0.6**: 分割面板

### 开发工具
- **ESLint 8.56.0**: 代码检查
- **Sass 1.69.5**: CSS预处理器
- **Vite Plugin SVG Icons 2.0.1**: SVG图标处理

## 项目结构

```
web/vue/
├── public/                      # 静态资源
│   ├── config.ini              # API配置文件(不提交)
│   ├── config.ini.example      # API配置示例
│   ├── favicon.ico             # 网站图标
│   └── index.html              # HTML模板
├── src/
│   ├── api/                    # API接口
│   │   └── request.js          # Axios封装
│   ├── assets/                 # 资源文件
│   │   ├── images/             # 图片资源
│   │   └── styles/             # 样式文件
│   ├── components/             # 公共组件
│   │   ├── Breadcrumb/         # 面包屑导航
│   │   ├── Charts/             # 图表组件
│   │   ├── CountTo/            # 数字动画
│   │   ├── DndList/            # 拖拽列表
│   │   ├── DragSelect/         # 拖拽选择
│   │   ├── Dropzone/           # 文件上传
│   │   ├── ErrorLog/           # 错误日志
│   │   ├── Hamburger/          # 汉堡菜单
│   │   ├── JsonEditor/         # JSON编辑器
│   │   ├── Kanban/             # 看板
│   │   ├── MarkdownEditor/     # Markdown编辑器
│   │   ├── Pagination/         # 分页组件
│   │   ├── PanThumb/           # 图片缩略图
│   │   ├── RightPanel/         # 右侧面板
│   │   ├── Share/              # 分享组件
│   │   ├── SizeSelect/         # 尺寸选择
│   │   ├── Sticky/             # 粘性布局
│   │   ├── SvgIcon/            # SVG图标
│   │   ├── TextHoverEffect/    # 文字悬停效果
│   │   ├── ThemePicker/        # 主题选择
│   │   ├── Tinymce/            # 富文本编辑器
│   │   └── Upload/             # 上传组件
│   ├── icons/                  # SVG图标
│   ├── layout/                 # 布局组件
│   │   ├── components/         # 布局子组件
│   │   │   ├── AppMain.vue     # 主内容区
│   │   │   ├── Navbar.vue      # 顶部导航栏
│   │   │   ├── Settings/       # 设置面板
│   │   │   ├── Sidebar/        # 侧边栏
│   │   │   └── TagsView/       # 标签页
│   │   └── index.vue           # 布局主组件
│   ├── router/                 # 路由配置
│   │   └── index.js            # 路由定义
│   ├── store/                  # Vuex状态管理
│   │   ├── modules/            # 状态模块
│   │   └── index.js            # Store入口
│   ├── styles/                 # 全局样式
│   │   ├── element-ui.scss     # Element Plus样式
│   │   ├── index.scss          # 主样式
│   │   ├── sidebar.scss        # 侧边栏样式
│   │   └── variables.scss      # 样式变量
│   ├── utils/                  # 工具函数
│   │   ├── auth.js             # 认证工具
│   │   ├── index.js            # 通用工具
│   │   ├── permission.js       # 权限控制
│   │   ├── request.js          # 请求工具
│   │   └── validate.js         # 表单验证
│   ├── views/                  # 页面组件
│   │   ├── attck/              # ATT&CK相关页面
│   │   │   ├── AttckMatrix.vue         # ATT&CK矩阵
│   │   │   ├── TechniqueDetail.vue     # 技术详情
│   │   │   ├── TechniqueMapping.vue    # 技术映射
│   │   │   ├── ApiComponentMapping.vue # API组件映射
│   │   │   ├── FunctionDetail.vue      # 函数详情
│   │   │   └── AttackPlanGenerator.vue # 攻击计划生成
│   │   ├── dashboard/          # 仪表盘
│   │   │   ├── sample.vue      # 样本统计
│   │   │   ├── domain.vue      # 域名统计
│   │   │   └── domain-map.vue  # 域名地图
│   │   │       ├── admin/      # 管理视图
│   │   │       └── editor/     # 编辑视图
│   │   ├── detect/             # 检测页面
│   │   │   ├── sample-vt.vue           # 样本检测
│   │   │   ├── domain.vue              # 域名检测
│   │   │   ├── model.vue               # 模型检测
│   │   │   ├── components/             # 检测组件
│   │   │   │   ├── AVDetection.vue     # 杀毒软件检测
│   │   │   │   ├── DynamicDetection.vue # 动态检测
│   │   │   │   ├── FileInfo.vue        # 文件信息
│   │   │   │   └── ModelDetection.vue  # 模型检测
│   │   │   └── documentation/          # 文档
│   │   ├── flowviz/            # 攻击流可视化
│   │   │   ├── index.vue               # 主页面
│   │   │   ├── FlowAnalysis.vue        # 流程分析
│   │   │   ├── FlowDebug.vue           # 调试页面
│   │   │   ├── FlowHistory.vue         # 历史记录
│   │   │   └── components/             # FlowViz组件
│   │   │       ├── FlowVisualization.vue    # 流程可视化
│   │   │       ├── FlowControlPanel.vue      # 控制面板
│   │   │       ├── FlowProgress.vue          # 进度显示
│   │   │       ├── FlowNodeDetails.vue       # 节点详情
│   │   │       ├── EdgeDetailPanel.vue       # 边详情
│   │   │       └── FlowHistory.vue           # 历史记录
│   │   ├── file_search/        # 文件搜索
│   │   │   ├── SHA256.vue      # SHA256搜索
│   │   │   ├── category.vue    # 分类搜索
│   │   │   ├── family.vue      # 家族搜索
│   │   │   ├── platform.vue    # 平台搜索
│   │   │   └── components/     # 搜索组件
│   │   ├── login/              # 登录页面
│   │   │   ├── index.vue       # 登录表单
│   │   │   ├── register.vue    # 注册表单
│   │   │   └── components/     # 登录组件
│   │   ├── settings/           # 设置页面
│   │   │   └── SystemSettings.vue  # 系统设置
│   │   ├── error-page/         # 错误页面
│   │   │   ├── 401.vue         # 401未授权
│   │   │   └── 404.vue         # 404未找到
│   │   ├── error-log/          # 错误日志
│   │   │   ├── index.vue       # 日志主页面
│   │   │   └── components/     # 日志组件
│   │   └── redirect/           # 重定向
│   │       └── index.vue       # 重定向页面
│   ├── App.vue                 # 根组件
│   ├── main.js                 # 应用入口
│   ├── permission.js           # 路由权限控制
│   └── settings.js             # 全局设置
├── .env.development            # 开发环境配置(不提交)
├── .env.production             # 生产环境配置(不提交)
├── .env.staging                # 测试环境配置(不提交)
├── .env.example                # 环境配置示例
├── .editorconfig               # 编辑器配置
├── .eslintrc.js                # ESLint配置
├── .eslintignore               # ESLint忽略文件
├── .gitignore                  # Git忽略文件
├── babel.config.js             # Babel配置
├── jest.config.js              # Jest配置
├── jsconfig.json               # JS配置
├── vite.config.js              # Vite配置
├── package.json                # 项目依赖
├── package-lock.json           # 依赖锁定文件
├── postcss.config.js           # PostCSS配置
└── README.md                   # 项目说明
```

## 快速开始

### 环境要求

- Node.js >= 14.0.0
- npm >= 6.0.0

### 安装步骤

1. **克隆项目**
```bash
git clone <repository_url>
cd nkrepo/web/vue
```

2. **安装依赖**
```bash
npm install
```

3. **配置环境变量**

复制环境变量示例文件:
```bash
cp .env.example .env.development
cp .env.example .env.production
cp .env.example .env.staging
```

编辑环境变量文件,配置API地址:
```bash
# .env.development
ENV = 'development'
VITE_APP_BASE_API = 'http://127.0.0.1:5005'

# .env.production
ENV = 'production'
VITE_APP_BASE_API = 'http://your-server-ip:5005'

# .env.staging
ENV = 'staging'
VITE_APP_BASE_API = 'http://your-staging-ip:5005'
```

**注意**: 后端启动时会自动更新这些文件中的API地址。

4. **配置API文件**

复制API配置示例文件:
```bash
cp public/config.ini.example public/config.ini
```

编辑配置文件:
```ini
[api]
baseUrl = http://127.0.0.1:5005
prefix = /api
```

5. **启动开发服务器**
```bash
npm run dev
```

访问: http://localhost:9528

6. **构建生产版本**
```bash
# 生产环境
npm run build:prod

# 测试环境
npm run build:stage

# 预览构建结果
npm run preview
```

## 主要功能

### 1. 样本检测
- **文件上传**: 支持拖拽上传和点击上传
- **多引擎检测**: 集成多种检测引擎
- **VirusTotal集成**: 支持VT检测
- **实时结果**: 实时显示检测进度和结果
- **历史记录**: 查看历史检测结果
- **分布式杀毒扫描**:
  - 单文件扫描: 支持多虚拟机并行扫描
  - 批量扫描: 支持多文件批量上传和扫描
  - 实时进度: 显示上传进度、扫描进度、当前文件名
  - 自动统计: 安全文件数、恶意文件数、扫描结果汇总

### 2. ATT&CK矩阵分析
- **企业矩阵视图**: 展示完整的ATT&CK矩阵
- **技术详情**: 查看每个技术的详细信息
- **技术映射**: 管理技术映射关系
- **API组件映射**: 查看API组件映射
- **函数详情分析**: 支持C++代码高亮
- **攻击计划生成**: 基于ATT&CK生成攻击计划

### 3. 攻击流可视化 (FlowViz)
- **流程图可视化**: 使用Dagre和D3.js渲染流程图
- **节点详情**: 查看节点的详细信息
- **边详情**: 查看边的详细信息
- **历史记录**: 保存和管理分析历史
- **导出功能**: 支持JSON、PNG、ATT&CK Flow v3、STIX格式导出
- **流式分析**: 实时显示AI分析过程
- **调试模式**: 查看分析过程的详细信息

### 4. 数据可视化
- **样本统计图表**: 恶意样本统计
- **域名分析**: 域名分布和趋势
- **攻击趋势分析**: 时间序列分析
- **分类统计**: 按Category、Platform、Family分类统计
- **域名地图**: 地理位置分布

### 5. 样本检索
- **SHA256搜索**: 按SHA256哈希搜索
- **分类搜索**: 按分类搜索
- **家族搜索**: 按家族搜索
- **平台搜索**: 按平台搜索
- **结果导出**: 支持导出搜索结果

### 6. 系统设置
- **系统配置**: 系统参数配置
- **用户认证**: JWT令牌认证

## 环境配置

### 开发环境配置

```bash
# .env.development
ENV = 'development'
VITE_APP_BASE_API = 'http://127.0.0.1:5005'
```

### 生产环境配置

```bash
# .env.production
ENV = 'production'
VITE_APP_BASE_API = 'http://your-server-ip:5005'
```

### 测试环境配置

```bash
# .env.staging
ENV = 'staging'
VITE_APP_BASE_API = 'http://your-staging-ip:5005'
```

### API代理配置

开发环境下,Vite会自动代理API请求:

```javascript
// vite.config.js
export default defineConfig({
  server: {
    proxy: {
      '/dev-api': {
        target: 'http://127.0.0.1:5005',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://127.0.0.1:5005',
        changeOrigin: true,
      },
      '/flowviz': {
        target: 'http://127.0.0.1:5005',
        changeOrigin: true,
      }
    }
  }
})
```

**注意**: 实际的API地址由`.env`文件中的`VITE_APP_BASE_API`变量控制。

## 开发指南

### 添加新页面

1. 在`src/views/`创建新的页面组件
2. 在`src/router/index.js`中添加路由配置
3. 在`src/layout/components/Sidebar/`中添加菜单项(如需要)

```javascript
// src/router/index.js
{
  path: '/new-page',
  component: Layout,
  children: [
    {
      path: 'index',
      name: 'NewPage',
      component: () => import('@/views/new-page/index'),
      meta: { title: '新页面', icon: 'example' }
    }
  ]
}
```

### 添加新组件

1. 在`src/components/`创建新的组件
2. 在需要的地方导入并使用

```vue
<template>
  <my-component />
</template>

<script>
import MyComponent from '@/components/MyComponent'

export default {
  components: { MyComponent }
}
</script>
```

### 添加API接口

1. 在`src/api/`创建API文件
2. 使用封装好的request.js发送请求

```javascript
// src/api/myApi.js
import request from '@/utils/request'

export function getData(params) {
  return request({
    url: '/api/my-endpoint',
    method: 'get',
    params
  })
}

export function postData(data) {
  return request({
    url: '/api/my-endpoint',
    method: 'post',
    data
  })
}
```

### 状态管理

使用Vuex进行状态管理:

```javascript
// src/store/modules/myModule.js
const state = {
  data: []
}

const mutations = {
  SET_DATA(state, data) {
    state.data = data
  }
}

const actions = {
  async fetchData({ commit }) {
    const data = await getData()
    commit('SET_DATA', data)
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
}
```

### 样式开发

使用Sass进行样式开发:

```scss
// src/styles/component.scss
.my-component {
  .header {
    font-size: 16px;
    color: $primary-color;
  }

  .content {
    padding: 20px;
  }
}
```

## Vue2到Vue3迁移说明

本项目已从Vue2成功迁移到Vue3,主要变更:

### 1. 构建工具
- Vue CLI → Vite
- Webpack → Vite内置打包器
- 配置文件: `vue.config.js` → `vite.config.js`

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
- `beforeDestroy` → `beforeUnmount`
- `destroyed` → `unmounted`

### 4. Composition API

推荐使用Composition API:

```vue
<script setup>
import { ref, computed, onMounted } from 'vue'

const count = ref(0)
const doubleCount = computed(() => count.value * 2)

onMounted(() => {
  console.log('Component mounted')
})

function increment() {
  count.value++
}
</script>
```

### 5. 配置文件
- `config.ini` → `.env`文件
- 环境变量使用 `VITE_` 前缀

详细的迁移说明请参考Vue3官方文档。

## 浏览器支持

- Chrome (推荐)
- Firefox
- Safari
- Edge

不支持IE浏览器。

## 性能优化

### 1. 路由懒加载
```javascript
const NewPage = () => import('@/views/new-page/index')
```

### 2. 组件懒加载
```javascript
const MyComponent = defineAsyncComponent(() =>
  import('@/components/MyComponent')
)
```

### 3. 图片优化
- 使用WebP格式
- 图片懒加载
- 压缩图片大小

### 4. 代码分割
- 使用动态导入
- 配置Vite的build.rollupOptions

### 5. 缓存策略
- 使用HTTP缓存
- 利用浏览器缓存

## 常见问题

### 1. 依赖安装失败
```bash
# 清除缓存重新安装
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### 2. 开发服务器启动失败
检查端口9528是否被占用,或修改`vite.config.js`中的端口号:
```javascript
export default defineConfig({
  server: {
    port: 9528
  }
})
```

### 3. API请求失败
- 检查后端服务是否启动
- 检查`.env`文件中的API地址是否正确
- 检查浏览器控制台的网络请求
- 检查CORS配置

### 4. 构建失败
```bash
# 清理缓存重新构建
rm -rf dist
npm run build
```

### 5. 样式不生效
- 检查Sass依赖是否安装
- 检查样式文件路径是否正确
- 检查Element Plus样式是否引入

### 6. 图标不显示
- 检查SVG图标配置
- 检查图标组件是否正确引入
- 检查图标路径是否正确

## 测试

```bash
# 运行单元测试
npm run test

# 运行E2E测试
npm run test:e2e
```

## 代码规范

项目使用ESLint进行代码检查:

```bash
# 检查代码
npm run lint

# 自动修复
npm run lint:fix
```

## 部署

### 1. 构建生产版本
```bash
npm run build:prod
```

### 2. 部署到服务器

将`dist`目录部署到Web服务器:

```bash
# 使用Nginx
cp -r dist/* /var/www/html/
```

### 3. 配置Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://127.0.0.1:5005;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /flowviz {
        proxy_pass http://127.0.0.1:5005;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 更新日志

### v2.0.0 (2026-03-11)
- 从Vue2迁移到Vue3
- 从Vue CLI迁移到Vite
- 从Element UI迁移到Element Plus
- 实现FlowViz流式分析界面
- 优化ATT&CK矩阵展示
- 改进样本检测界面
- 添加域名地图可视化
- 完善文档和配置示例

### v1.0.0 (2025-02-28)
- 初始版本
- Vue2前端实现
- 基础功能开发

## 贡献指南

欢迎提交Issue和Pull Request!

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

MIT License
