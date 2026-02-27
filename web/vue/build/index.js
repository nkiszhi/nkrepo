// vue/src/router/index.js
import Vue from 'vue'
import Router from 'vue-router'
import Layout from '@/layout'

Vue.use(Router)

export const constantRoutes = [
  {
    path: '/redirect',
    component: Layout,
    hidden: true,
    children: [
      {
        path: '/redirect/:path(.*)',
        component: () => import('@/views/redirect/index')
      }
    ]
  },
  {
    path: '/login',
    component: () => import('@/views/login/index'),
    hidden: true
  },
  {
    path: '/404',
    component: () => import('@/views/error-page/404'),
    hidden: true
  },
  {
    path: '/',
    component: Layout,
    redirect: '/sample',
    alwaysShow: true,
    name: 'Dashboard',
    meta: { title: '样本库', icon: 'dashboard' },
    children: [
      {
        path: 'sample',
        component: () => import('@/views/dashboard/sample'),
        name: 'Sample',
        meta: { title: '恶意文件样本数据展示', affix: true }
      },
      {
        path: 'domain-map',
        component: () => import('@/views/dashboard/domain-map'),
        name: 'DomainMap',
        meta: { title: '恶意域名样本数据展示', affix: true }
      }
    ]
  }
]

export const asyncRoutes = [
  {
    path: '/detect',
    component: Layout,
    redirect: '/detect/sample-vt',
    name: 'Detect',
    meta: { title: '样本检测', icon: 'detect' },
    children: [
      {
        path: 'sample-vt',
        component: () => import('@/views/detect/sample-vt'),
        name: 'SampleVt',
        meta: { title: '恶意文件检测' }
      },
      {
        path: 'domain',
        component: () => import('@/views/detect/domain'),
        name: 'DomainDetect',
        meta: { title: '恶意域名检测' }
      }
    ]
  },
  {
    path: '/file_search',
    component: Layout,
    redirect: '/file_search/category',
    name: 'FileSearch',
    meta: { title: '样本检索', icon: 'search' },
    children: [
      {
        path: 'SHA256',
        component: () => import('@/views/file_search/SHA256'),
        name: 'SHA256Search',
        meta: { title: 'SHA256检索' }
      },
      {
        path: 'category',
        component: () => import('@/views/file_search/category'),
        name: 'CategorySearch',
        hidden: true,
        meta: { title: '类型检索' }
      },
      {
        path: 'family',
        component: () => import('@/views/file_search/family'),
        name: 'FamilySearch',
        meta: { title: '家族检索' }
      },
      {
        path: 'platform',
        component: () => import('@/views/file_search/platform'),
        name: 'PlatformSearch',
        hidden: true,
        meta: { title: '平台检索' }
      }
    ]
  },
  {
    path: '/flowviz',
    component: Layout,
    redirect: '/flowviz/analysis',
    name: 'FlowViz',
    meta: { title: '攻击流分析', icon: 'chart' },
    children: [
      {
        path: 'analysis',
        component: () => import('@/views/flowviz/FlowAnalysis'),
        name: 'FlowVizAnalysis',
        meta: { title: '流式分析' }
      },
      {
        path: 'history',
        component: () => import('@/views/flowviz/FlowHistory'),
        name: 'FlowHistory',
        meta: { title: '分析历史' }
      }
    ]
  },
  {
    path: '/attck',
    component: Layout,
    redirect: '/attck/matrix',
    name: 'ATT&CK',
    meta: { title: 'ATT&CK 矩阵', icon: 'table' },
    children: [
      {
        path: 'matrix',
        component: () => import('@/views/attck/AttckMatrix'),
        name: 'AttckMatrix',
        meta: { title: '企业矩阵视图' }
      },
      {
        path: 'api-components',
        component: () => import('@/views/attck/ApiComponentMapping'),
        name: 'ApiComponentMapping',
        meta: { title: 'API组件映射' } // 新增的界面
      },
      {
        path: 'technique/:id',
        component: () => import('@/views/attck/TechniqueDetail'),
        name: 'TechniqueDetail',
        hidden: true, // 详情页不显示在侧边栏
        meta: { title: '技术详情' }
      },
      {
        path: 'technique-mapping',
        component: () => import('@/views/attck/TechniqueMapping'),
        name: 'TechniqueMapping',
        meta: { title: '技术映射管理' }
      },
      {
        path: 'attack-plan',
        component: () => import('@/views/attck/AttackPlanGenerator'),
        name: 'AttackPlanGenerator',
        meta: { title: '攻击方案生成' }
      }
    ]
  },
  {
    path: '/settings',
    component: Layout,
    redirect: '/settings/system',
    name: 'Settings',
    meta: { title: '系统设置', icon: 'setting' },
    children: [
      {
        path: 'system',
        component: () => import('@/views/settings/SystemSettings'),
        name: 'SystemSettings',
        meta: { title: '系统配置' }
      }
    ]
  },
  { path: '*', redirect: '/404', hidden: true }
]

const router = new Router({
  scrollBehavior: () => ({ y: 0 }),
  routes: constantRoutes
})

export function resetRouter() {
  const newRouter = new Router({
    scrollBehavior: () => ({ y: 0 }),
    routes: constantRoutes
  })
  router.matcher = newRouter.matcher
}

export default router
