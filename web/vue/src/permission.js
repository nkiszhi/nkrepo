import router from './router/index.js'
import store from './store/index.js'
import { ElMessage } from 'element-plus'
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'
import { getToken } from '@/utils/auth'
import getPageTitle from '@/utils/get-page-title'

NProgress.configure({ showSpinner: false })

const whiteList = ['/login', '/auth-redirect']

router.beforeEach(async(to, from, next) => {
  NProgress.start()
  document.title = getPageTitle(to.meta.title)

  const hasToken = getToken() || localStorage.getItem('token') || sessionStorage.getItem('token')

  if (hasToken) {
    if (to.path === '/login') {
      next({ path: '/' })
      NProgress.done()
    } else {
      const hasRoles = store.getters.roles && store.getters.roles.length > 0
      if (hasRoles) {
        next()
      } else {
        try {
          const roles = ['admin']
          store.commit('user/SET_ROLES', roles)

          const accessRoutes = await store.dispatch('permission/generateRoutes', roles)

          accessRoutes.forEach(route => {
            router.addRoute(route)
          })

          next({ ...to, replace: true })
        } catch (error) {
          await store.dispatch('user/resetToken')
          ElMessage.error(error.message || '登录状态失效，请重新登录')
          next(`/login?redirect=${to.path}`)
          NProgress.done()
        }
      }
    }
  } else {
    if (whiteList.indexOf(to.path) !== -1) {
      next()
    } else {
      next(`/login?redirect=${to.path}`)
      NProgress.done()
    }
  }
})

router.afterEach(() => {
  NProgress.done()
})
