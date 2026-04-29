import request from '@/utils/request'

/**
 * 代码同源性检测相关API
 */

/**
 * 分析代码文件相似度
 * @param {FormData} data - 包含代码文件的FormData对象
 * @returns {Promise} 分析结果
 */
export function analyzeCode(data) {
  return request({
    url: '/api/dolos/analyze',
    method: 'post',
    data,
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    timeout: 300000 // 5分钟超时
  })
}

/**
 * 获取分析历史记录
 * @param {Object} params - 查询参数
 * @param {number} params.skip - 跳过记录数
 * @param {number} params.limit - 返回记录数
 * @param {string} params.search - 搜索关键词
 * @param {string} params.start_date - 开始日期
 * @param {string} params.end_date - 结束日期
 * @returns {Promise} 历史记录列表
 */
export function getAnalysisHistory(params) {
  return request({
    url: '/api/dolos/history',
    method: 'get',
    params
  })
}

/**
 * 获取特定分析结果
 * @param {string} analysisId - 分析ID
 * @returns {Promise} 分析结果详情
 */
export function getAnalysisResult(analysisId) {
  return request({
    url: `/api/dolos/result/${analysisId}`,
    method: 'get'
  })
}

/**
 * 删除分析结果
 * @param {string} analysisId - 分析ID
 * @returns {Promise} 删除结果
 */
export function deleteAnalysisResult(analysisId) {
  return request({
    url: `/api/dolos/result/${analysisId}`,
    method: 'delete'
  })
}

/**
 * 获取支持的文件类型列表
 * @returns {Promise} 文件类型列表
 */
export function getSupportedFileTypes() {
  return request({
    url: '/api/dolos/file-types',
    method: 'get'
  })
}

/**
 * 批量分析URL中的代码文件
 * @param {Object} data - 包含文件URL列表的对象
 * @param {Array<string>} data.urls - 文件URL列表
 * @returns {Promise} 分析结果
 */
export function analyzeFromUrls(data) {
  return request({
    url: '/api/dolos/analyze-urls',
    method: 'post',
    data
  })
}
