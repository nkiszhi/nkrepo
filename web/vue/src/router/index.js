import { createRouter, createWebHistory } from 'vue-router'
import Layout from '@/layout/index.vue'

export const constantRoutes = [
  {
    path: '/redirect',
    component: Layout,
    hidden: true,
    children: [
      {
        path: '/redirect/:path(.*)',
        component: () => import('@/views/redirect/index.vue')
      }
    ]
  },
  {
    path: '/login',
    component: () => import('@/views/login/index.vue'),
    hidden: true
  },
  {
    path: '/404',
    component: () => import('@/views/error-page/404.vue'),
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
        component: () => import('@/views/dashboard/sample.vue'),
        name: 'Sample',
        meta: { title: '恶意文件样本数据展示', affix: true }
      },
      {
        path: 'function-detail',
        component: () => import('@/views/attck/FunctionDetail.vue'),
        name: 'FunctionDetail',
        hidden: true,
        meta: { title: '函数详情', affix: true }
      },
      {
        path: 'domain-map',
        component: () => import('@/views/dashboard/domain-map.vue'),
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
        component: () => import('@/views/detect/sample-vt.vue'),
        name: 'SampleVt',
        meta: { title: '恶意文件检测' }
      },
      {
        path: 'domain',
        component: () => import('@/views/detect/domain.vue'),
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
        component: () => import('@/views/file_search/SHA256.vue'),
        name: 'SHA256Search',
        meta: { title: 'SHA256检索' }
      },
      {
        path: 'category',
        component: () => import('@/views/file_search/category.vue'),
        name: 'CategorySearch',
        hidden: true,
        meta: { title: '类型检索' }
      },
      {
        path: 'family',
        component: () => import('@/views/file_search/family.vue'),
        name: 'FamilySearch',
        meta: { title: '家族检索' }
      },
      {
        path: 'platform',
        component: () => import('@/views/file_search/platform.vue'),
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
        component: () => import('@/views/flowviz/FlowAnalysis.vue'),
        name: 'FlowVizAnalysis',
        meta: { title: '流式分析' }
      },
      {
        path: 'history',
        component: () => import('@/views/flowviz/FlowHistory.vue'),
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
        component: () => import('@/views/attck/AttckMatrix.vue'),
        name: 'AttckMatrix',
        meta: { title: '企业矩阵视图' }
      },
      {
        path: 'api-components',
        component: () => import('@/views/attck/ApiComponentMapping.vue'),
        name: 'ApiComponentMapping',
        meta: { title: 'API组件映射' }
      },
      {
        path: 'technique/:id',
        component: () => import('@/views/attck/TechniqueDetail.vue'),
        name: 'TechniqueDetail',
        hidden: true,
        meta: { title: '技术详情' }
      },
      {
        path: 'technique-mapping',
        component: () => import('@/views/attck/TechniqueMapping.vue'),
        name: 'TechniqueMapping',
        meta: { title: '技术映射管理' }
      },
      {
        path: 'function-detail',
        component: () => import('@/views/attck/FunctionDetail.vue'),
        name: 'FunctionDetail',
        hidden: true,
        meta: { title: '函数详情' }
      },
      {
        path: 'attack-plan',
        component: () => import('@/views/attck/AttackPlanGenerator.vue'),
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
    meta: { title: '系统设置', icon: 'component' },
    children: [
      {
        path: 'system',
        component: () => import('@/views/settings/SystemSettings.vue'),
        name: 'SystemSettings',
        meta: { title: '系统配置' }
      }
    ]
  },
  { path: '/:pathMatch(.*)*', redirect: '/404', hidden: true }
]

const router = createRouter({
  history: createWebHistory(),
  scrollBehavior: () => ({ top: 0 }),
  routes: constantRoutes
})

export function resetRouter() {
  const newRouter = createRouter({
    history: createWebHistory(),
    scrollBehavior: () => ({ top: 0 }),
    routes: constantRoutes
  })
  router.matcher = newRouter.matcher
}

export default router
