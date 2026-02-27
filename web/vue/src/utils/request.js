import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import store from '@/store'
import { getToken } from '@/utils/auth'

const service = axios.create({
  baseURL: '',
  timeout: 60000,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json'
  }
})

const staticService = axios.create({
  baseURL: '/',
  timeout: 30000,
  headers: {
    'Cache-Control': 'max-age=3600'
  }
})

const isDevEnv = import.meta.env.MODE === 'development'
const currentPath = window.location.pathname
const isDevApiPath = currentPath.includes('/dev-api/')

function getBaseUrl() {
  if (isDevEnv) {
    return ''
  }
  if (currentPath.includes('/flowviz/')) {
    return ''
  }
  return ''
}

service.interceptors.request.use(
  config => {
    console.log('🔧 请求配置:', {
      url: config.url,
      baseURL: config.baseURL,
      method: config.method
    })

    config.baseURL = getBaseUrl()

    if (config.url && config.url.includes('/flowviz/')) {
      console.log('🎯 FlowViz请求，使用相对路径:', config.url)
      if (config.url.startsWith('/api/flowviz/')) {
        config.url = config.url.replace('/api/flowviz/', '/flowviz/')
        console.log('🔄 修正路径为:', config.url)
      }
    } else if (config.url && !config.url.startsWith('http')) {
      if (!config.url.startsWith('/api/') &&
          !config.url.startsWith('/dev-api/') &&
          !config.url.startsWith('/flowviz/')) {
        if (isDevApiPath) {
          config.url = '/dev-api' + (config.url.startsWith('/') ? config.url : '/' + config.url)
          console.log('🔄 自动添加 /dev-api 前缀:', config.url)
        } else {
          config.url = '/api' + (config.url.startsWith('/') ? config.url : '/' + config.url)
          console.log('🔄 自动添加 /api 前缀:', config.url)
        }
      }
    }

    const token = getToken() || localStorage.getItem('token') || localStorage.getItem('flowviz_token')

    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`
      console.log('✅ 已添加Authorization头')
    } else {
      console.log('⚠️ 没有可用的Token')
    }

    if (config.url && config.url.includes('/flowviz/')) {
      config.headers['Accept'] = 'application/json'
      config.headers['Cache-Control'] = 'no-cache'
      config.headers['X-Requested-With'] = 'XMLHttpRequest'

      if (config.url.includes('analyze-stream')) {
        config.timeout = 300000
        config.headers['Accept'] = 'text/event-stream'
      }
    }

    if (config.url && config.url.includes('/attck/')) {
      config.headers['X-Requested-With'] = 'XMLHttpRequest'
    }

    return config
  },
  error => {
    console.error('请求配置错误:', error)
    return Promise.reject(error)
  }
)

staticService.interceptors.request.use(
  config => {
    console.log('📁 请求静态文件:', config.url)
    if (config.url && config.url.endsWith('.json')) {
      config.headers['Cache-Control'] = 'max-age=3600, public'
    }
    return config
  },
  error => {
    console.error('静态文件请求配置错误:', error)
    return Promise.reject(error)
  }
)

const createResponseInterceptor = (isStatic = false) => {
  return (response) => {
    const res = response.data

    console.log('✅ 请求成功:', {
      url: response.config.url,
      status: response.status,
      data: typeof res
    })

    if (response.headers['content-type'] &&
        response.headers['content-type'].includes('text/event-stream')) {
      return response
    }

    if (isStatic) {
      return res
    }

    if (res.success === false) {
      ElMessage({
        message: res.message || 'Error',
        type: 'error',
        duration: 5 * 1000
      })
      return Promise.reject(new Error(res.message || 'Error'))
    } else {
      return res
    }
  }
}

const createErrorHandler = (isStatic = false) => {
  return (error) => {
    const serviceType = isStatic ? '静态文件' : 'API'
    console.error(`${serviceType}请求错误:`, error)

    if (error.response) {
      const status = error.response.status
      const data = error.response.data
      const url = error.config.url

      console.error('❌ 请求错误详情:', {
        url: url,
        method: error.config.method,
        status: status,
        data: data
      })

      if (status === 401 && !isStatic) {
        localStorage.removeItem('token')
        localStorage.removeItem('flowviz_token')
        ElMessageBox.confirm(
          '登录状态已过期，请重新登录',
          '确认登出',
          {
            confirmButtonText: '重新登录',
            cancelButtonText: '取消',
            type: 'warning'
          }
        ).then(() => {
          store.dispatch('user/resetToken').then(() => {
            location.reload()
          })
        })
      } else if (status === 403) {
        ElMessage({
          message: '禁止访问',
          type: 'error',
          duration: 5 * 1000
        })
      } else if (status === 404) {
        ElMessage({
          message: `请求的资源不存在 (404): ${url}`,
          type: 'error',
          duration: 5 * 1000
        })

        if (url.includes('/flowviz/')) {
          console.error('❌ FlowViz API未找到，请检查后端路由注册:')
          console.error('   - 确保app.py中注册了FlowViz蓝图')
          console.error('   - 确保routes/providers.py文件存在')
          console.error('   - 检查Flask应用是否正常运行')
        }
      } else if (status === 500) {
        ElMessage({
          message: '服务器内部错误',
          type: 'error',
          duration: 5 * 1000
        })
      } else {
        const errorMsg = data?.message || data?.error || error.message
        ElMessage({
          message: `请求失败 (${status}): ${errorMsg}`,
          type: 'error',
          duration: 5 * 1000
        })
      }
    } else if (error.code === 'ECONNABORTED') {
      ElMessage({
        message: '请求超时，请稍后重试',
        type: 'error',
        duration: 5 * 1000
      })
    } else if (error.message === 'Network Error') {
      ElMessage({
        message: '网络连接失败，请检查网络设置',
        type: 'error',
        duration: 5 * 1000
      })
    } else {
      ElMessage({
        message: `请求错误: ${error.message}`,
        type: 'error',
        duration: 5 * 1000
      })
    }

    return Promise.reject(error)
  }
}

service.interceptors.response.use(
  createResponseInterceptor(false),
  createErrorHandler(false)
)

staticService.interceptors.response.use(
  createResponseInterceptor(true),
  createErrorHandler(true)
)

export { service, staticService }
export default service
