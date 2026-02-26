import { createRouter, createWebHashHistory } from 'vue-router'

/* Layout */
import Layout from '@/layout'

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
    path: '/auth-redirect',
    component: () => import('@/views/login/auth-redirect'),
    hidden: true
  },
  {
    path: '/404',
    component: () => import('@/views/error-page/404'),
    hidden: true
  },
  {
    path: '/401',
    component: () => import('@/views/error-page/401'),
    hidden: true
  },
  {
    path: '/',
    component: Layout,
    redirect: '/sample',
    alwaysShow: true,
    name: 'Dashboard',
    meta: {
      title: '样本库',
      icon: 'dashboard',
      roles: ['admin', 'editor'] // you can set roles in root nav
    },
    children: [
      {
        path: 'sample',
        component: () => import('@/views/dashboard/sample'),
        name: 'sample',
        meta: { title: '恶意文件样本数据展示', affix: true }
      },
      {
        path: 'domain-map',
        component: () => import('@/views/dashboard/domain-map'),
        name: 'domain-map',
        meta: { title: '恶意域名样本数据展示', affix: true }
      }
    ]
  },
  {
    path: '/profile',
    component: Layout,
    redirect: '/profile/index',
    hidden: true,
    children: [
      {
        path: 'index',
        component: () => import('@/views/profile/index'),
        name: 'Profile',
        meta: { title: 'Profile', icon: 'user', noCache: true }
      }
    ]
  }
]

export const asyncRoutes = [
  /**
   * asyncRoutes
   * the routes that need to be dynamically loaded based on user roles
   */
  {
    path: '/detect',
    component: Layout,
    redirect: '/detect/sample',
    name: 'Detect',
    meta: {
      title: '样本检测',
      icon: 'detect'
    },
    children: [
      {
        path: 'sample-vt',
        component: () => import('@/views/detect/sample-vt'),
        name: 'detect_sample-vt',
        meta: { title: '恶意文件检测' }
      },
      {
        path: 'domain',
        component: () => import('@/views/detect/domain'),
        name: 'detect_domain',
        meta: { title: '恶意域名检测' }
      }
    ]
  },

  {
    path: '/file_search',
    component: Layout,
    redirect: '/file_search/category',
    name: 'FileSearch',
    meta: {
      title: '样本检索',
      icon: 'search'
    },
    children: [
      {
        path: 'SHA256',
        component: () => import('@/views/file_search/SHA256'),
        name: 'SHA256',
        meta: { title: 'SHA256' }
      },
      {
        path: 'category',
        component: () => import('@/views/file_search/category'),
        name: 'category',
        meta: { title: '类型检索' }
      },
      {
        path: 'family',
        component: () => import('@/views/file_search/family'),
        name: 'family',
        meta: { title: '家族检索' }
      },
      {
        path: 'platform',
        component: () => import('@/views/file_search/platform'),
        name: 'platform',
        meta: { title: '平台检索' }
      }
    ]
  },

  // 404 page must be placed at the end !!!
  { path: '/:pathMatch(.*)*', redirect: '/404', hidden: true }
]

const router = createRouter({
  history: createWebHashHistory(),
  scrollBehavior: () => ({ top: 0 }),
  routes: constantRoutes
})

// Detail see: https://github.com/vuejs/vue-router/issues/1234#issuecomment-357941465
export function resetRouter() {
  const newRouter = createRouter({
    history: createWebHashHistory(),
    scrollBehavior: () => ({ top: 0 }),
    routes: constantRoutes
  })
  // Reset router matcher
  router.matcher = newRouter.matcher
}

export default router
