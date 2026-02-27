import axios from 'axios'
import { ElMessageBox, ElMessage } from 'element-plus'
import store from '@/store'
import { getToken } from '@/utils/auth'

// 直接使用环境变量的baseUrl（已修改为后端地址）
const service = axios.create({
  baseURL: process.env.VUE_APP_BASE_API, // 现在是 http://10.134.13.242:5005
  timeout: 10000 // 延长超时时间
})

// request interceptor
service.interceptors.request.use(
  config => {
    if (store.getters.token) {
      // 关键：后端要求的Bearer Token格式
      config.headers['Authorization'] = `Bearer ${getToken()}`
    }
    return config
  },
  error => {
    console.log(error) // for debug
    return Promise.reject(error)
  }
)

// response interceptor
service.interceptors.response.use(
  response => {
    const res = response.data
    return res // 直接返回后端数据，不额外校验
  },
  error => {
    console.log('err' + error) // for debug
    const errMsg = error.response?.data?.error || error.message || '请求失败'
    ElMessage({
      message: errMsg,
      type: 'error',
      duration: 5 * 1000
    })

    // Token过期处理
    if (error.response && error.response.status === 401) {
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
    }
    return Promise.reject(error)
  }
)

export default service
