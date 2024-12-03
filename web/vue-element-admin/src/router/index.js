import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

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
      // {
      //   path: 'dashboard',
      //   component: () => import('@/views/dashboard/index'),
      //   name: 'Dashboard',
      //   meta: { title: 'Dashboard', affix: true }
      // },
      {
        path: 'sample',
        component: () => import('@/views/dashboard/sample'),
        name: 'sample',
        meta: { title: '恶意文件样本数据展示',affix: true }
      },
      {
        path: 'domain',
        component: () => import('@/views/dashboard/domain'),
        name: 'domain',
        meta: { title: '恶意域名样本数据展示',affix: true }
      },
      {
        path: 'domain-map',
        component: () => import('@/views/dashboard/domain-map'),
        name: 'domain-map',
        meta: { title: '恶意域名样本数据展示-地图',affix: true }
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
        path: 'sample',
        component: () => import('@/views/detect/sample'),
        name: 'detect_sample',
        meta: { title: '恶意文件检测' }
      },
     
      {
        path: 'domain',
        component: () => import('@/views/detect/domain'),
        name: 'detect_domain',
        meta: { title: '恶意域名检测' }
      },
      {
        path: 'domain-map',
        component: () => import('@/views/detect/domain-map'),
        name: 'detect_domain',
        meta: { title: '恶意域名检测-地图' }
      },
      {
        path: 'model',
        component: () => import('@/views/detect/model'),
        name: 'detect_model',
        meta: { title: '模型检测' }
      }
    ]
  },
  
  {
    path: '/file_search',
    component: Layout,
    redirect: '/file_search/category',
    name: 'Search',
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
  { path: '*', redirect: '/404', hidden: true }
]

const createRouter = () => new Router({
  // mode: 'history', // require service support
  scrollBehavior: () => ({ y: 0 }),
  routes: constantRoutes
})

const router = createRouter()

// Detail see: https://github.com/vuejs/vue-router/issues/1234#issuecomment-357941465
export function resetRouter() {
  const newRouter = createRouter()
  router.matcher = newRouter.matcher // reset router
}

export default router
