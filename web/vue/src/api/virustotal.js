import request from '@/utils/request'

/**
 * Get VirusTotal detection results for a sample
 * @param {string} sha256 - SHA256 hash of the sample (64 hex characters)
 */
export function getDetectionResults(sha256) {
  return request({
    url: `/detection_API/${sha256}`,
    method: 'get'
  })
}

/**
 * Get VirusTotal behaviour analysis results for a sample
 * @param {string} sha256 - SHA256 hash of the sample (64 hex characters)
 */
export function getBehaviourResults(sha256) {
  return request({
    url: `/behaviour_API/${sha256}`,
    method: 'get'
  })
}
