import Vue from 'vue'
import Router from 'vue-router'
import store from '@/store'
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
    path: '/register',
    component: () => import('@/views/login/register'),
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
        meta: { title: '个人中心', icon: 'user', noCache: true }
      }
    ]
  }
]

export const asyncRoutes = [
  {
    path: '/detect',
    component: Layout,
    redirect: '/detect/sample',
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
        meta: { title: '平台检索' }
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


router.beforeEach((to, from, next) => {
  console.log('路由守卫触发:', from.path, '->', to.path)
  
  // 从Vuex或本地存储获取登录状态
  const hasToken = store.getters.token || localStorage.getItem('token') || sessionStorage.getItem('token')
  
  // 关键修复：直接允许访问/register，忽略所有查询参数
  if (to.path === '/register') {
    console.log('允许直接访问注册页，忽略查询参数')
    next()
    return
  }

  if (hasToken) {
    console.log('用户已登录，token存在')
    
    if (to.path === '/login' || to.path === '/register') {
      // 已登录时访问登录页或注册页，重定向到首页
      console.log('已登录，重定向到首页')
      next({ path: '/' })
    } else {
      // 允许访问其他页面
      next()
    }
  } else {
    console.log('用户未登录，token不存在')
    
    if (to.path === '/login' || to.path === '/register') {
      // 允许访问登录页和注册页
      next()
    } else {
      // 重定向到登录页，并携带原目标路径参数
      console.log('未登录，重定向到登录页，原路径:', to.fullPath)
      next(`/login?redirect=${to.fullPath}`)
    }
  }
})

export default router