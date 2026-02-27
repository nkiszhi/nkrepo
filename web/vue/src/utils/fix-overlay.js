// src/utils/fix-overlay.js

/**
 * 修复 Element UI 遮罩层问题
 */
export function fixElementOverlays() {
  try {
    // 移除遮罩层
    const overlays = document.querySelectorAll('.el-overlay, .el-loading-mask, .v-modal')
    overlays.forEach(overlay => {
      if (overlay && overlay.parentNode === document.body) {
        overlay.style.display = 'none'
        overlay.style.opacity = '0'
        overlay.style.zIndex = '-9999'
        
        // 尝试移除
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
    
    // 修复 body 样式
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

/**
 * 确保页面可交互
 */
export function ensurePageInteractive() {
  // 修复遮罩层
  fixElementOverlays()
  
  // 确保所有元素可交互
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
 * 路由守卫 - 在路由变化时修复页面
 */
export function createRouterGuard(router) {
  router.beforeEach((to, from, next) => {
    // 在路由变化前修复页面
    fixElementOverlays()
    next()
  })
  
  router.afterEach(() => {
    // 在路由变化后确保页面可交互
    setTimeout(ensurePageInteractive, 100)
  })
}