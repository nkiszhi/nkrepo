import SvgIcon from '@/components/SvgIcon/index.vue'

// 使用Vite的import.meta.glob来导入所有SVG文件
const svgModules = import.meta.glob('./svg/*.svg', { eager: true })

// 导出注册函数
export function registerIcons(app) {
  // 注册SvgIcon组件
  app.component('svg-icon', SvgIcon)
  
  // 注册所有SVG图标
  Object.keys(svgModules).forEach(key => {
    const fileName = key.replace(/\.\/svg\/(.*)\.svg$/, '$1')
    // SVG文件会被vite-plugin-svg-icons处理
  })
}

export default { registerIcons }
