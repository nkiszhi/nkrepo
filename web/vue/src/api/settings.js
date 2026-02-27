// vue/src/api/settings.js
import request from '@/utils/request'

// 获取所有配置
export function getConfigs() {
  return request({
    url: '/api/config/all',
    method: 'get'
  })
}

// 更新配置
export function updateConfigs(data) {
  return request({
    url: '/api/config/update',
    method: 'post',
    data
  })
}
