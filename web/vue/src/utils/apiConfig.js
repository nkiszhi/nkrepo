/**
 * API配置工具 - Vue3风格
 * 统一管理API地址,从环境变量读取
 */

// 从环境变量获取API地址
const getBaseApi = () => {
  // Vite环境变量,以VITE_开头
  const baseApi = import.meta.env.VITE_APP_BASE_API
  
  if (!baseApi) {
    console.warn('⚠️ 未找到VITE_APP_BASE_API环境变量,使用默认地址')
    return 'http://10.134.53.143:5005'
  }
  
  return baseApi
}

// 导出API地址
export const BASE_API = getBaseApi()

// 导出配置对象
export default {
  BASE_API,
  
  // API端点
  get API_LOGIN() { return `${BASE_API}/api/login` },
  get API_DETECT() { return `${BASE_API}/api/detect` },
  get API_SEARCH() { return `${BASE_API}/api/search` },
  get API_FAMILY() { return `${BASE_API}/api/family` },
  get API_DOWNLOAD() { return `${BASE_API}/api/download` },
  
  // 流式分析API
  get FLOWVIZ_API() { return `${BASE_API}/flowviz/api` },
  
  // 打印配置信息
  log() {
    console.log('🚀 API配置:')
    console.log('  BASE_API:', BASE_API)
    console.log('  环境:', import.meta.env.MODE)
  }
}
