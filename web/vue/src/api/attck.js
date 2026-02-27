// vue/src/api/attck.js (完整版)
import request from '@/utils/request'
import { staticService } from '@/utils/request'

// 缓存静态矩阵数据
let cachedMatrixData = null

/**
 * ATT&CK矩阵核心接口
 */
export default {
  // 1. 获取ATT&CK矩阵静态结构数据
  async getAttckMatrix() {
    // 如果已有缓存数据,直接返回
    if (cachedMatrixData) {
      console.log('📦 从缓存返回矩阵数据')
      return cachedMatrixData
    }

    console.log('🔄 开始加载ATT&CK矩阵数据...')

    try {
      // 方案1: 尝试从assets目录import(推荐)
      try {
        const module = await import('@/assets/matrix-enterprise.json')
        cachedMatrixData = module.default
        console.log('✅ 从assets目录加载矩阵数据成功')
        return cachedMatrixData
      } catch (importError) {
        console.warn('从assets目录加载失败,尝试方案2:', importError)

        // 方案2: 尝试从public目录请求
        try {
          const response = await staticService.get('/matrix-enterprise.json')
          cachedMatrixData = response
          console.log('✅ 从public目录加载矩阵数据成功')
          return cachedMatrixData
        } catch (staticError) {
          console.warn('从public目录加载失败,使用方案3:', staticError)

          // 方案3: 使用模拟数据
          console.log('⚠️ 使用模拟矩阵数据')
          cachedMatrixData = this.getMockMatrixData()
          return cachedMatrixData
        }
      }
    } catch (error) {
      console.error('加载矩阵数据失败,使用模拟数据:', error)
      cachedMatrixData = this.getMockMatrixData()
      return cachedMatrixData
    }
  },

  // 2. 获取ATT&CK技术列表(带函数统计的)
  getTechniquesList(params) {
    return request({
      url: '/dev-api/api/attck/techniques',
      method: 'get',
      params: {
        page: params?.page || 1,
        page_size: params?.pageSize || 20,
        ...params
      }
    })
  },

  // 3. 获取技术详情(后端API)
  getTechniqueDetail(techniqueId) {
    return request({
      url: `/dev-api/api/attck/techniques/${techniqueId}`,
      method: 'get'
    })
  },
  
  // 5. 获取矩阵统计数据
  getMatrixStats() {
    return request({
      url: '/dev-api/api/attck/matrix/stats',
      method: 'get'
    })
  },

  // 6. 获取统计信息(用于顶部卡片)
  getStatistics() {
    return request({
      url: '/dev-api/api/attck/statistics',
      method: 'get'
    })
  },

  // 7. 搜索ATT&CK技术
  searchAttck(keyword) {
    return request({
      url: '/dev-api/api/attck/search',
      method: 'get',
      params: { keyword }
    })
  },

  // 8. 代码分析接口
  analyzeCode(data) {
    return request({
      url: '/dev-api/api/analysis/code',
      method: 'post',
      data
    })
  },

  // 9. 创建攻击方案接口
  createAttackPlan(data) {
    return request({
      url: '/dev-api/api/analysis/attack-plan',
      method: 'post',
      data
    })
  },

  // 10. 获取函数列表(用于代码分析)
  getFunctions(params) {
    return request({
      url: '/dev-api/api/functions',
      method: 'get',
      params: {
        page: params?.page || 1,
        page_size: params?.pageSize || 20,
        ...params
      }
    })
  },

  // 11. 获取函数详情
  getFunctionDetail(functionId) {
    return request({
      url: `/dev-api/api/functions/${functionId}`,
      method: 'get'
    })
  },

  // 12. 获取战术详情
  getTacticDetail(tacticId) {
    return request({
      url: `/dev-api/api/attck/tactic/${tacticId}`,
      method: 'get'
    })
  },

  // 13. 清除矩阵数据缓存(用于开发调试)
  clearMatrixCache() {
    cachedMatrixData = null
    console.log('🗑️ 已清除矩阵数据缓存')
  },

  // 14. 模拟数据(备用)
  getMockMatrixData() {
    console.log('🎭 使用模拟矩阵数据')
    return {
      isMock: true,
      'TA0043': {
        'tactic_name_en': 'Reconnaissance',
        'tactic_name_cn': '侦察',
        'techniques': [
          {
            'T1595': 'Active Scanning',
            'sub': [
              { 'T1595.001': 'Scanning IP Blocks' },
              { 'T1595.002': 'Vulnerability Scanning' },
              { 'T1595.003': 'Wordlist Scanning' }
            ]
          },
          {
            'T1592': 'Gather Victim Host Information',
            'sub': [
              { 'T1592.001': 'Hardware' },
              { 'T1592.002': 'Software' },
              { 'T1592.003': 'Firmware' },
              { 'T1592.004': 'Client Configurations' }
            ]
          },
          {
            'T1589': 'Gather Victim Identity Information',
            'sub': [
              { 'T1589.001': 'Credentials' },
              { 'T1589.002': 'Email Addresses' },
              { 'T1589.003': 'Employee Names' }
            ]
          }
        ]
      },
      'TA0042': {
        'tactic_name_en': 'Resource Development',
        'tactic_name_cn': '资源开发',
        'techniques': [
          {
            'T1583': 'Acquire Infrastructure',
            'sub': [
              { 'T1583.001': 'Domains' },
              { 'T1583.002': 'DNS Server' },
              { 'T1583.003': 'Virtual Private Server' },
              { 'T1583.004': 'Server' },
              { 'T1583.005': 'Botnet' },
              { 'T1583.006': 'Web Services' }
            ]
          },
          {
            'T1586': 'Compromise Accounts',
            'sub': [
              { 'T1586.001': 'Social Media Accounts' },
              { 'T1586.002': 'Email Accounts' },
              { 'T1586.003': 'Cloud Accounts' }
            ]
          }
        ]
      },
      'TA0001': {
        'tactic_name_en': 'Initial Access',
        'tactic_name_cn': '初始访问',
        'techniques': [
          {
            'T1078': 'Valid Accounts',
            'sub': [
              { 'T1078.001': 'Default Accounts' },
              { 'T1078.002': 'Domain Accounts' },
              { 'T1078.003': 'Local Accounts' },
              { 'T1078.004': 'Cloud Accounts' }
            ]
          },
          {
            'T1566': 'Phishing',
            'sub': [
              { 'T1566.001': 'Spearphishing Attachment' },
              { 'T1566.002': 'Spearphishing Link' },
              { 'T1566.003': 'Spearphishing via Service' }
            ]
          }
        ]
      },
      'TA0002': {
        'tactic_name_en': 'Execution',
        'tactic_name_cn': '执行',
        'techniques': [
          {
            'T1059': 'Command and Scripting Interpreter',
            'sub': [
              { 'T1059.001': 'PowerShell' },
              { 'T1059.002': 'AppleScript' },
              { 'T1059.003': 'Windows Command Shell' },
              { 'T1059.004': 'Unix Shell' },
              { 'T1059.005': 'Visual Basic' },
              { 'T1059.006': 'Python' },
              { 'T1059.007': 'JavaScript' },
              { 'T1059.008': 'Network Device CLI' }
            ]
          }
        ]
      }
    }
  },

  // 15. 获取API组件映射列表
  getApiComponents(params) {
    return request({
      url: '/dev-api/api/attck/api-components',
      method: 'get',
      params: {
        page: params?.page || 1,
        page_size: params?.pageSize || 20,
        search: params?.search,
        ...params
      }
    })
  },

  // 16. 获取API组件详情
  getApiComponentDetail(hashId, apiComponent) {
    return request({
      url: '/dev-api/api/attck/api-component/detail',
      method: 'get',
      params: {
        hash_id: hashId,
        api_component: apiComponent
      }
    })
  },

  // 17. 搜索API组件
  searchApiComponents(keyword, page = 1, pageSize = 20) {
    return request({
      url: '/dev-api/api/attck/api-components',
      method: 'get',
      params: {
        search: keyword,
        page: page,
        page_size: pageSize
      }
    })
  },
  
  // 18. 获取技术映射列表
  getTechniqueMapping(params) {
    return request({
      url: '/dev-api/api/attck/technique-mapping',
      method: 'get',
      params: {
        page: params?.page || 1,
        page_size: params?.pageSize || 20,
        search: params?.search,
        ...params
      }
    })
  },

  // 19. 获取技术对应的函数列表
  getTechniqueFunctions(techniqueId) {
    return request({
      url: '/dev-api/api/attck/technique-functions',
      method: 'get',
      params: {
        technique_id: techniqueId
      }
    })
  },

  // 20. 获取技术对应的函数列表(详细版)
  getTechniqueFunctionsDetail(techniqueId, params) {
    return request({
      url: '/dev-api/api/attck/function/list',
      method: 'get',
      params: {
        technique_id: techniqueId,
        page: params?.page || 1,
        page_size: params?.pageSize || 10,
        ...params
      }
    })
  },

  // 21. 获取函数详情(包含C++源代码)
  getFunctionDetail(params) {
    return request({
      url: '/dev-api/api/attck/function/detail',
      method: 'get',
      params: params
    })
  }
}