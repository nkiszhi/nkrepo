<template>
  <main>
    <div class="text-center">
      <h2 class="text-primary">Codefender 多模型恶意文件检测</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG - 支持33个检测模型</p>
    </div>

    <!-- 检测方式切换 -->
    <div class="detection-mode-container">
      <el-radio-group v-model="detectionMode" @change="handleModeChange">
        <el-radio-button label="file">文件检测</el-radio-button>
        <el-radio-button label="sha256">SHA256检测</el-radio-button>
      </el-radio-group>
    </div>

    <!-- 文件检测模式 -->
    <div v-if="detectionMode === 'file'" class="detection-content">
      <input ref="file-upload-input" class="file-upload-input" type="file" @change="handleFileChange">
      <div class="drop" @drop="handleDrop" @dragover="handleDragover">
        把待检文件拖到这里或
        <el-button
          :loading="loading"
          style="margin-left:0%;font-size: 20px;"
          size="small"
          type="primary"
          @click="handleUpload"
        >
          选择待检文件
        </el-button>
      </div>
    </div>

    <!-- SHA256检测模式 -->
    <div v-else class="detection-content">
      <div class="sha256-input-container">
        <el-input
          v-model="sha256Input"
          placeholder="请输入文件的SHA256哈希值"
          clearable
          :disabled="loading"
          class="sha256-input"
          @keyup.enter="handleSha256Submit"
          @input="handleSha256Input"
        >
          <template #prepend>
            <svg-icon icon-class="hash" style="width: 20px; height: 20px;" />
          </template>
          <template #append>
            <div class="button-group">
              <el-button
                type="default"
                :disabled="loading && !sha256Input"
                title="重置"
                @click="handleReset"
              >
                <svg-icon icon-class="reset" style="width: 16px; height: 16px; margin-right: 4px;" />
                重置
              </el-button>
              <el-button
                :type="detectButtonType"
                :loading="loading"
                :disabled="!isValidSha256 || loading"
                :class="{ 'highlight-button': isValidSha256 }"
                title="开始检测"
                @click="handleSha256Submit"
              >
                <svg-icon icon-class="search" style="width: 16px; height: 16px; margin-right: 4px;" />
                检测
              </el-button>
            </div>
          </template>
        </el-input>

        <!-- 字符计数和验证提示 -->
        <div class="input-validation">
          <div class="char-count">
            已输入: <span :class="charCountClass">{{ sha256Input.length }}</span> / 64 字符
          </div>
          <div v-if="sha256Input.length > 0 && sha256Input.length < 64" class="validation-hint">
            <i class="el-icon-warning" style="color: #E6A23C; margin-right: 8px;" />
            <span>请输入完整的64位SHA256哈希值</span>
          </div>
          <div v-else-if="sha256Input.length === 64 && !isValidSha256" class="validation-hint">
            <i class="el-icon-circle-close" style="color: #F56C6C; margin-right: 8px;" />
            <span>SHA256格式不正确，必须为64位十六进制字符 (0-9, a-f)</span>
          </div>
          <div v-else-if="isValidSha256" class="validation-hint success">
            <i class="el-icon-circle-check" style="color: #67C23A; margin-right: 8px;" />
            <span>SHA256格式正确，可以开始检测</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 检测结果区域 -->
    <div v-if="uploadResult">
      <div style="text-align: center;" class="result-content">
        <svg-icon style="width: 30px; height: 30px;margin-right: 10px;margin-top:30px; " icon-class="detect-report" />
        <span class="result-status">检测报告如下</span>

        <!-- 显示检测模式 -->
        <div class="detection-mode-tag">
          <el-tag :type="detectionMode === 'file' ? 'primary' : 'success'">
            {{ detectionMode === 'file' ? '文件检测' : 'SHA256检测' }}
          </el-tag>
          <el-tag v-if="detectionMode === 'sha256'" type="info" style="margin-left: 8px;">
            SHA256: {{ formatSha256(sha256Input) }}
          </el-tag>
          <el-tag v-if="ensembleResult" :type="ensembleResult.result === '恶意' ? 'danger' : 'success'" style="margin-left: 8px;">
            多模型结果：{{ ensembleResult.result }}
          </el-tag>
          <el-tag v-if="ensembleResult && ensembleResult.virus_name" type="warning" style="margin-left: 8px;">
            病毒名称：{{ ensembleResult.virus_name }}
          </el-tag>
        </div>
      </div>

      <!-- 导航按钮 -->
      <div class="nav-buttons-container">
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'fileInfo' }"
          @click="showSection = 'fileInfo'"
        >
          <svg-icon icon-class="fileInfo" class="button-icon" />
          <span>基础信息</span>
        </button>
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'modelDetection' }"
          @click="showSection = 'modelDetection'"
        >
          <svg-icon icon-class="modelDetection" class="button-icon" />
          <span>模型检测</span>
        </button>
      </div>

      <!-- 基础信息区域 -->
      <div v-show="showSection === 'fileInfo'" class="section">
        <table class="info-table">
          <tbody>
            <tr>
              <td class="label">文件名</td>
              <td class="value">{{ uploadResult.filename || '-' }}</td>
            </tr>
            <tr>
              <td class="label">SHA256</td>
              <td class="value sha256-value">{{ uploadResult.sha256 || '-' }}</td>
            </tr>
            <tr>
              <td class="label">文件大小</td>
              <td class="value">{{ formatFileSize(uploadResult.file_size) }}</td>
            </tr>
            <tr>
              <td class="label">文件类型</td>
              <td class="value">{{ uploadResult.file_type || '-' }}</td>
            </tr>
            <tr>
              <td class="label">检测时间</td>
              <td class="value">{{ uploadResult.detection_time || '-' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- 模型检测结果区域 - 33个模型 -->
      <div v-show="showSection === 'modelDetection'" class="section">
        <div class="model-summary">
          <el-tag type="info" size="large">共 {{ modelCount }} 个检测模型</el-tag>
          <el-tag type="danger" size="large" style="margin-left: 10px;">恶意: {{ maliciousCount }}</el-tag>
          <el-tag type="success" size="large" style="margin-left: 10px;">安全: {{ safeCount }}</el-tag>
        </div>
        
        <table class="detection-result-table">
          <thead>
            <tr>
              <th style="width: 60px;">序号</th>
              <th>检测模型</th>
              <th style="width: 150px;">恶意概率</th>
              <th style="width: 100px;">结果</th>
              <th style="width: 220px;">病毒名称</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(resultData, model, index) in modelResults" :key="model">
              <td>{{ index + 1 }}</td>
              <td>{{ model }}</td>
              <td>
                <el-progress 
                  :percentage="getProbabilityPercent(resultData.probability)" 
                  :color="getProbabilityColor(resultData.probability)"
                  :stroke-width="18"
                />
              </td>
              <td>
                <el-tag :type="resultData.result === '恶意' ? 'danger' : 'success'" size="small">
                  {{ resultData.result }}
                </el-tag>
              </td>
              <td>{{ resultData.virus_name || '-' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>
</template>

<script>
import axios from 'axios'
import { ElMessage } from 'element-plus'

// 创建axios实例
const apiService = axios.create({
  timeout: 600000,
  headers: { 'Content-Type': 'application/json' }
})

// 请求拦截器
apiService.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token') || sessionStorage.getItem('token')
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  error => Promise.reject(error)
)

export default {
  name: 'CodefenderDetect',

  data() {
    return {
      detectionMode: 'file',
      sha256Input: '',
      showSection: 'fileInfo',
      loading: false,
      uploadResult: null,
      apiBaseUrl: 'http://10.134.13.242:5005' // 系统后端默认地址
    }
  },

  computed: {
    // SHA256格式验证
    isValidSha256() {
      const sha256Regex = /^[a-fA-F0-9]{64}$/
      return sha256Regex.test(this.sha256Input.trim())
    },

    // 检测按钮类型
    detectButtonType() {
      return this.isValidSha256 ? 'primary' : 'default'
    },

    // 字符计数样式
    charCountClass() {
      if (this.sha256Input.length === 64) return 'char-count-valid'
      if (this.sha256Input.length > 64) return 'char-count-exceed'
      return 'char-count-normal'
    },

    // 模型结果
    modelResults() {
      const results = this.uploadResult?.exe_result || this.uploadResult?.results || {}
      return Object.fromEntries(
        Object.entries(results).filter(([model]) => model !== '集成结果' && model !== 'error')
      )
    },

    // 集成结果
    ensembleResult() {
      const results = this.uploadResult?.exe_result || this.uploadResult?.results || {}
      return results['集成结果'] || null
    },

    // 模型数量
    modelCount() {
      return Object.keys(this.modelResults).length
    },

    // 恶意数量
    maliciousCount() {
      return Object.values(this.modelResults).filter(r => r.result === '恶意').length
    },

    // 安全数量
    safeCount() {
      return Object.values(this.modelResults).filter(r => r.result === '安全').length
    }
  },

  created() {
    this.loadConfig()
  },

  methods: {
    async loadConfig() {
      try {
        const response = await apiService.get('/config.ini', {
          responseType: 'text',
          timeout: 5000
        })
        const lines = response.data.split('\n')
        let currentSection = ''
        let apiBaseUrl = ''
        let codefenderBaseUrl = ''
        for (const line of lines) {
          const trimmedLine = line.trim()
          if (!trimmedLine || trimmedLine.startsWith('#')) continue
          if (trimmedLine.startsWith('[') && trimmedLine.endsWith(']')) {
            currentSection = trimmedLine.slice(1, -1).toLowerCase()
            continue
          }
          if ((currentSection === 'api' || currentSection === 'codefender') && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=')
            if (parts.length >= 2) {
              if (currentSection === 'api') apiBaseUrl = parts[1].trim()
              if (currentSection === 'codefender') codefenderBaseUrl = parts[1].trim()
            }
          }
        }
        this.apiBaseUrl = codefenderBaseUrl || apiBaseUrl || this.apiBaseUrl
      } catch (error) {
        console.warn('加载配置文件失败:', error.message)
      }
    },

    handleModeChange() {
      this.uploadResult = null
      this.sha256Input = ''
    },

    handleSha256Input() {
      this.sha256Input = this.sha256Input.toLowerCase()
    },

    handleReset() {
      this.sha256Input = ''
      this.uploadResult = null
      ElMessage.info('已重置')
    },

    formatSha256(sha256) {
      if (!sha256 || sha256.length < 64) return sha256
      return `${sha256.substring(0, 12)}...${sha256.substring(52)}`
    },

    formatFileSize(bytes) {
      if (!bytes) return '-'
      if (typeof bytes === 'string') return bytes
      const k = 1024
      const sizes = ['B', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
    },

    getProbabilityPercent(probability) {
      if (typeof probability === 'string') {
        return parseFloat(probability.replace('%', '')) || 0
      }
      return (probability * 100) || 0
    },

    getProbabilityColor(probability) {
      const percent = this.getProbabilityPercent(probability)
      if (percent >= 70) return '#F56C6C'
      if (percent >= 40) return '#E6A23C'
      return '#67C23A'
    },

    handleDrop(e) {
      if (this.detectionMode !== 'file') return
      e.stopPropagation()
      e.preventDefault()
      if (this.loading) return
      const files = e.dataTransfer.files
      if (files.length !== 1) {
        ElMessage.error('只支持上传一个文件!')
        return
      }
      this.uploadFile(files[0])
    },

    handleDragover(e) {
      if (this.detectionMode !== 'file') return
      e.stopPropagation()
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    },

    handleUpload() {
      if (this.detectionMode !== 'file') return
      this.$refs['file-upload-input'].click()
    },

    handleFileChange(e) {
      if (this.detectionMode !== 'file') return
      const files = e.target.files
      if (files[0]) this.uploadFile(files[0])
    },

    handleSha256Submit() {
      if (!this.isValidSha256) {
        ElMessage.error('请输入有效的64位SHA256哈希值')
        return
      }
      this.detectBySha256(this.sha256Input.trim().toLowerCase())
    },

    async uploadFile(rawFile) {
      this.uploadResult = null
      this.loading = true

      const formData = new FormData()
      formData.append('file', rawFile)

      try {
        const response = await apiService.post(`${this.apiBaseUrl}/detect`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 600000
        })
        this.uploadResult = this.normalizeDetectionResponse(response.data)
        ElMessage.success('检测完成')
      } catch (error) {
        console.error('检测失败:', error)
        ElMessage.error('检测失败: ' + (error.response?.data?.detail || error.message))
      } finally {
        this.loading = false
      }
    },

    async detectBySha256(sha256) {
      this.uploadResult = null
      this.loading = true

      try {
        const response = await apiService.post(`${this.apiBaseUrl}/detect_by_sha256`, {
          sha256: sha256
        }, { timeout: 600000 })
        this.uploadResult = this.normalizeDetectionResponse(response.data)
        ElMessage.success('检测完成')
      } catch (error) {
        console.error('检测失败:', error)
        ElMessage.error('检测失败: ' + (error.response?.data?.detail || error.message))
      } finally {
        this.loading = false
      }
    },

    normalizeDetectionResponse(data) {
      const queryResult = data.query_result || {}
      return {
        ...data,
        filename: data.filename || data.original_filename || queryResult.name || '-',
        sha256: data.sha256 || queryResult['SHA-256'] || queryResult.SHA256 || '',
        file_size: data.file_size || queryResult['文件大小'] || 0,
        file_type: data.file_type || queryResult['文件类型'] || queryResult['类型'] || '-',
        detection_time: data.detection_time || new Date().toLocaleString(),
        exe_result: data.exe_result || data.results || {}
      }
    }
  }
}
</script>

<style scoped>
.file-upload-input {
  display: none;
}

.detection-mode-container {
  text-align: center;
  margin: 20px 0;
}

.detection-content {
  margin: 20px auto;
  width: 80%;
  max-width: 800px;
}

.drop {
  border: 2px dashed #bbb;
  width: 100%;
  height: 160px;
  line-height: 160px;
  margin: 0 auto;
  font-size: 24px;
  border-radius: 5px;
  text-align: center;
  color: #bbb;
}

.sha256-input-container {
  margin-top: 20px;
}

.sha256-input {
  font-family: 'Courier New', monospace;
  font-size: 16px;
}

.button-group {
  display: flex;
  gap: 8px;
}

.highlight-button {
  background-color: #409eff !important;
  border-color: #409eff !important;
  color: white !important;
}

.input-validation {
  margin-top: 10px;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.char-count {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.char-count-valid { color: #67c23a; font-weight: bold; }
.char-count-exceed { color: #f56c6c; font-weight: bold; }
.char-count-normal { color: #909399; }

.validation-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.validation-hint.success { color: #67c23a; }

.detection-mode-tag {
  margin-top: 10px;
  text-align: center;
  display: flex;
  justify-content: center;
  align-items: center;
}

.result-status {
  font-size: 30px;
  color: #333;
  font-weight: bold;
  text-align: center;
  display: inline-block;
  margin-top: 20px;
}

.nav-buttons-container {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin: 30px auto;
  width: 60%;
}

.custom-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 200px;
  height: 50px;
  background-color: white;
  border: 1px solid #ccc;
  cursor: pointer;
  font-size: 16px;
  color: #333;
  transition: all 0.3s ease;
  border-radius: 4px;
}

.custom-button:hover {
  background-color: #f5f5f5;
}

.active-button {
  background-color: #409EFF;
  color: white;
  border-color: #409EFF;
}

.button-icon {
  width: 24px;
  height: 24px;
  margin-right: 8px;
}

.section {
  width: 80%;
  margin: 20px auto;
}

.info-table {
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
  border-collapse: collapse;
  background: white;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.info-table td {
  padding: 12px 20px;
  border-bottom: 1px solid #eee;
}

.info-table .label {
  width: 150px;
  font-weight: bold;
  color: #606266;
  background: #f5f7fa;
}

.info-table .value {
  color: #303133;
}

.sha256-value {
  font-family: 'Courier New', monospace;
  font-size: 13px;
  word-break: break-all;
}

.model-summary {
  text-align: center;
  margin-bottom: 20px;
}

.detection-result-table {
  width: 100%;
  border: 1px solid #ccc;
  border-collapse: collapse;
  background: white;
}

.detection-result-table th,
.detection-result-table td {
  padding: 12px 15px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.detection-result-table th {
  background: #f5f7fa;
  font-weight: bold;
  color: #606266;
}

.detection-result-table tbody tr:hover {
  background: #f5f7fa;
}
</style>
