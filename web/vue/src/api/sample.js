import request from '@/utils/request'

/**
 * Upload a file for malware detection analysis
 * @param {FormData} data - FormData containing the file
 */
export function uploadSample(data) {
  return request({
    url: '/upload',
    method: 'post',
    data,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

/**
 * Query samples by category, family, or platform
 * @param {string} searchType - One of: 'category', 'family', 'platform'
 * @param {string} tableName - The value to search for (e.g., 'trojan', 'wannacry', 'windows')
 */
export function querySamples(searchType, tableName) {
  return request({
    url: `/query/${searchType}`,
    method: 'post',
    data: { tableName }
  })
}

/**
 * Query sample by SHA256 hash
 * @param {string} sha256 - SHA256 hash of the sample
 */
export function querySampleBySha256(sha256) {
  return request({
    url: '/query_sha256',
    method: 'post',
    data: { tableName: sha256 }
  })
}

/**
 * Get sample details by SHA256 hash
 * @param {string} sha256 - SHA256 hash of the sample (64 hex characters)
 */
export function getSampleDetail(sha256) {
  return request({
    url: `/detail/${sha256}`,
    method: 'get'
  })
}

/**
 * Download sample as password-protected ZIP
 * @param {string} sha256 - SHA256 hash of the sample
 * Note: ZIP password is 'infected'
 */
export function downloadSample(sha256) {
  return request({
    url: `/download/${sha256}`,
    method: 'get',
    responseType: 'blob'
  })
}
