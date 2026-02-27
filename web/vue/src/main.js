import { createApp } from 'vue'
import Cookies from 'js-cookie'
import 'normalize.css/normalize.css'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import './styles/element-variables.scss'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import '@/styles/index.scss'

import App from './App.vue'
import store from './store/index.js'
import router from './router/index.js'
import { registerIcons } from './icons'
import 'virtual:svg-icons-register'  // 导入SVG图标注册
import './permission'
import { setupErrorHandler } from './utils/error-log.js'
import * as filters from './filters/index.js'

/**
 * 修复 Element Plus 遮罩层问题的工具函数
 */
const fixElementOverlays = () => {
  try {
    const overlays = document.querySelectorAll('.el-overlay, .el-loading-mask, .v-modal')
    overlays.forEach(overlay => {
      if (overlay && overlay.parentNode === document.body) {
        overlay.style.display = 'none'
        overlay.style.opacity = '0'
        overlay.style.zIndex = '-9999'
        setTimeout(() => {
          try {
            if (overlay.parentNode) {
              overlay.parentNode.removeChild(overlay)
            }
          } catch (e) {
            console.warn('移除遮罩层时出错:', e)
          }
        }, 50)
      }
    })
    const body = document.body
    const html = document.documentElement
    body.style.overflow = 'auto'
    body.style.position = 'static'
    body.style.width = 'auto'
    body.style.height = 'auto'
    body.style.paddingRight = '0'
    body.style.filter = 'none'
    body.style.opacity = '1'
    body.classList.remove('el-popup-parent--hidden')
    html.style.overflow = 'auto'
    return true
  } catch (error) {
    console.error('修复遮罩层失败:', error)
    return false
  }
}

const ensurePageInteractive = () => {
  fixElementOverlays()
  setTimeout(() => {
    try {
      const elements = document.querySelectorAll('*')
      elements.forEach(el => {
        if (el && el.style) {
          el.style.pointerEvents = 'auto'
          el.style.cursor = 'auto'
        }
      })
    } catch (error) {
      console.warn('设置元素交互性时出错:', error)
    }
  }, 100)
}

/**
 * 全局修复遮罩层的混入
 */
const overlayMixin = {
  mounted() {
    setTimeout(() => {
      fixElementOverlays()
    }, 100)
    setTimeout(() => {
      ensurePageInteractive()
    }, 300)
  },
  beforeUnmount() {
    if (this._fixTimer) {
      clearTimeout(this._fixTimer)
    }
  }
}

/**
 * 在路由变化时修复页面
 */
const createRouterGuard = (router) => {
  router.beforeEach((to, from, next) => {
    fixElementOverlays()
    next()
  })
  router.afterEach(() => {
    setTimeout(ensurePageInteractive, 100)
  })
}

/**
 * 监听页面变化，自动修复遮罩层
 */
const setupOverlayObserver = () => {
  const observer = new MutationObserver(() => {
    const overlays = document.querySelectorAll('.el-overlay, .el-loading-mask, .v-modal')
    overlays.forEach(overlay => {
      if (overlay && overlay.style && overlay.style.display !== 'none') {
        fixElementOverlays()
      }
    })
  })
  observer.observe(document.body, {
    childList: true,
    subtree: true
  })
}

const app = createApp(App)

// 注册SVG图标组件
registerIcons(app)

// 使用 Element Plus
app.use(ElementPlus, {
  size: Cookies.get('size') || 'medium',
  locale: zhCn,
})

// 设置错误处理
setupErrorHandler(app)

// 注册全局工具过滤器
Object.keys(filters).forEach(key => {
  app.config.globalProperties[`$${key}`] = filters[key]
})

// 使用路由和状态管理
app.use(store)
app.use(router)

// 注册全局混入
app.mixin(overlayMixin)

// 在应用启动前执行初始化修复
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(fixElementOverlays, 100)
  setTimeout(ensurePageInteractive, 300)
  setTimeout(setupOverlayObserver, 500)
})

// 创建路由守卫
createRouterGuard(router)

// 全局错误处理
app.config.errorHandler = function (err, vm, info) {
  console.error('Vue error:', err, info)
  setTimeout(fixElementOverlays, 100)
}

// 创建全局的页面恢复函数
window.$fixOverlays = fixElementOverlays
window.$ensureInteractive = ensurePageInteractive

app.mount('#app')

console.log('应用已启动，遮罩层修复系统已激活')

setTimeout(() => {
  fixElementOverlays()
  ensurePageInteractive()
}, 1000)
