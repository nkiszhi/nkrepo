<template>
  <main>
    <div class="text-center">
      <h2 class="text-primary">基于可信度评估的多模型恶意文件检测</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
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

        <div class="input-hint">
          <el-alert
            title="提示"
            type="info"
            :closable="false"
            show-icon
          >
            <p>请输入完整的64位SHA256哈希值，示例：</p>
            <code>7d5a3b8c9e2f1a4b6c8d0e2f3a5b7c9d1e3f4a6b8c0d2e4f5a7b9c1d3e5f6a8bc</code>
            <p class="hint-tips">支持大写和小写字母，系统会自动转换为小写</p>
          </el-alert>
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
          <el-button
            v-if="detectionMode === 'sha256'"
            type="text"
            size="small"
            style="margin-left: 8px;"
            title="复制SHA256"
            @click="copySha256"
          >
            <svg-icon icon-class="copy" style="width: 14px; height: 14px;" />
          </el-button>
        </div>
      </div>

      <!-- 导航按钮 -->
      <div
        style="display: flex; justify-content: space-between; align-items: center; margin-top:50px; width: 60%;margin: 0 auto"
      >
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'fileInfo' }"
          @click="showSection = 'fileInfo'"
        >
          <svg-icon icon-class="fileInfo" class="button-icon" />
          <span> 基础信息</span>
        </button>
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'modelDetection' }"
          @click="showSection = 'modelDetection'"
        >
          <svg-icon icon-class="modelDetection" class="button-icon" />
          <span> 模型检测</span>
        </button>
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'AV-Detection' }"
          @click="showSection = 'AV-Detection'"
        >
          <svg-icon icon-class="AV-Detection" class="button-icon" />
          <span> 杀软检测</span>
        </button>
        <button
          class="custom-button"
          :class="{ 'active-button': showSection === 'DynamicDetection' }"
          @click="showSection = 'DynamicDetection'"
        >
          <svg-icon icon-class="DynamicDetection" class="button-icon" />
          <span> 动态检测</span>
        </button>
      </div>

      <!-- 组件区域 -->
      <FileInfo
        v-show="showSection === 'fileInfo'"
        :upload-result="uploadResult"
      />

      <ModelDetection
        v-show="showSection === 'modelDetection'"
        :upload-result="uploadResult"
      />

      <AVDetection
        v-show="showSection === 'AV-Detection'"
        :results="results"
        :is-loading="isElLoading"
        :is-errors="isErrors"
        :valid-detector-count="validDetectorCount"
        :malicious-count="maliciousCount"
      />

      <DynamicDetection
        v-show="showSection === 'DynamicDetection'"
        :behaviour-results="behaviour_results"
        :is-loadings="isElLoadings"
        :is-error="isError"
        :has-valid-data="hasValidData"
      />
    </div>
  </main>
</template>

<script>
import axios from 'axios'
import FileInfo from './components/FileInfo.vue'
import ModelDetection from './components/ModelDetection.vue'
import AVDetection from './components/AVDetection.vue'
import DynamicDetection from './components/DynamicDetection.vue'

// 创建axios实例（设置10分钟超时，统一处理Token）
const apiService = axios.create({
  timeout: 600000, // 10分钟超时（600秒）
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器：自动携带Token
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
  name: 'SampleVT',
  components: {
    FileInfo,
    ModelDetection,
    AVDetection,
    DynamicDetection
  },

  data() {
    return {
      detectionMode: 'file', // 检测模式：file 或 sha256
      sha256Input: '', // SHA256输入值
      showSection: 'fileInfo',
      loading: false,
      isElLoading: false,
      isElLoadings: false,
      results: [], // 杀软检测结果
      behaviour_results: {}, // 动态检测结果，初始化为空对象
      uploadResult: null,
      isError: false,
      isErrors: false,
      apiBaseUrl: 'http://10.134.13.242:5005' // 默认API地址
    }
  },

  computed: {
    // SHA256格式验证（64位十六进制）
    isValidSha256() {
      const sha256Regex = /^[a-fA-F0-9]{64}$/
      return sha256Regex.test(this.sha256Input.trim())
    },

    // 检测按钮类型，当输入达到64位且格式正确时显示为primary
    detectButtonType() {
      return this.isValidSha256 ? 'primary' : 'default'
    },

    // 字符计数样式
    charCountClass() {
      if (this.sha256Input.length === 64) {
        return 'char-count-valid'
      } else if (this.sha256Input.length > 64) {
        return 'char-count-exceed'
      } else {
        return 'char-count-normal'
      }
    },

    hasValidData() {
      if (!this.behaviour_results || typeof this.behaviour_results !== 'object') {
        console.log('hasValidData: behaviour_results is invalid')
        return false
      }

      const r = this.behaviour_results
      console.log('hasValidData: checking behaviour_results:', r)

      // 检查是否有任何有效数据
      const validKeys = [
        'calls_highlighted', 'services_opened', 'services_started',
        'command_executions', 'files_attribute_changed', 'files_copied',
        'files_deleted', 'files_dropped', 'files_opened', 'files_written',
        'modules_loaded', 'mutexes_created', 'mutexes_opened',
        'permissions_requested', 'processes_terminated', 'processes_tree',
        'dns_lookups', 'http_conversations', 'ip_traffic', 'tls',
        'verdicts', 'attack_techniques', 'ids_alerts', 'mbc',
        'mitre_attack_techniques', 'signature_matches',
        'system_property_lookups',
        'memory_dumps', 'memory_pattern_domains', 'memory_pattern_urls',
        'registry_keys_deleted', 'registry_keys_opened', 'registry_keys_set',
        'crypto_algorithms_observed', 'crypto_plain_text', 'text_highlighted'
      ]

      for (const key of validKeys) {
        if (r[key] && Array.isArray(r[key]) && r[key].length > 0) {
          console.log(`hasValidData: found valid data in key "${key}":`, r[key].length)
          return true
        }
      }

      console.log('hasValidData: no valid data found')
      return false
    },

    // 杀软检测统计
    unsupportedCount() {
      return this.results.filter(item => item.category === 'type-unsupported').length
    },
    validDetectorCount() {
      return this.results.length - this.unsupportedCount
    },
    maliciousCount() {
      return this.results.filter(item =>
        item.category &&
        item.category !== 'type-unsupported' &&
        item.category !== 'undetected' &&
        item.category !== 'harmless'
      ).length
    },
    harmlessCount() {
      return this.validDetectorCount - this.maliciousCount
    }
  },

  created() {
    // 组件创建时加载配置文件
    this.loadConfig()
  },

  methods: {
    // 加载配置文件获取API地址
    async loadConfig() {
      try {
        const response = await apiService.get('/config.ini', {
          responseType: 'text',
          timeout: 5000 // 配置读取超时5秒
        })

        const configContent = response.data
        const lines = configContent.split('\n')
        let inApiSection = false

        for (const line of lines) {
          const trimmedLine = line.trim()
          if (trimmedLine === '[api]') {
            inApiSection = true
            continue
          }

          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=')
            if (parts.length >= 2) {
              this.apiBaseUrl = parts[1].trim()
              console.log('从配置文件加载API地址:', this.apiBaseUrl)
              break
            }
          }

          if (inApiSection && trimmedLine.startsWith('[')) {
            break
          }
        }

        // 配置文件未读取到baseUrl时，使用兜底地址
        if (!this.apiBaseUrl) {
          this.apiBaseUrl = 'http://10.134.13.242:5005'
          console.warn('配置文件未找到baseUrl，使用兜底地址')
        }
      } catch (error) {
        console.warn('加载配置文件失败，使用兜底地址:', error.message)
        this.apiBaseUrl = 'http://10.134.13.242:5005'
      }
    },

    // 处理检测模式切换
    handleModeChange() {
      // 清空之前的检测结果
      this.uploadResult = null
      this.results = []
      this.behaviour_results = {}
      this.sha256Input = ''
    },

    // 处理SHA256输入
    handleSha256Input() {
      // 自动将输入转换为小写
      this.sha256Input = this.sha256Input.toLowerCase()
    },

    // 重置按钮处理
    handleReset() {
      this.sha256Input = ''
      this.uploadResult = null
      this.results = []
      this.behaviour_results = {}
      this.isError = false
      this.isErrors = false
      this.$message.info('已重置输入和检测结果')
    },

    // 格式化SHA256显示
    formatSha256(sha256) {
      if (!sha256 || sha256.length < 64) return sha256
      return `${sha256.substring(0, 12)}...${sha256.substring(52)}`
    },

    // 复制SHA256到剪贴板
    copySha256() {
      this.copyText(this.sha256Input)
    },

    // 复制文本到剪贴板
    copyText(text) {
      navigator.clipboard.writeText(text).then(() => {
        this.$message.success('已复制到剪贴板')
      }).catch(err => {
        console.error('复制失败:', err)
        // 降级方案
        const textarea = document.createElement('textarea')
        textarea.value = text
        document.body.appendChild(textarea)
        textarea.select()
        try {
          document.execCommand('copy')
          this.$message.success('已复制到剪贴板')
        } catch (e) {
          this.$message.error('复制失败')
        }
        document.body.removeChild(textarea)
      })
    },

    handleDrop(e) {
      if (this.detectionMode !== 'file') return

      e.stopPropagation()
      e.preventDefault()
      if (this.loading) return
      const files = e.dataTransfer.files
      if (files.length !== 1) {
        this.$message.error('只支持上传一个文件!')
        return
      }
      const rawFile = files[0]
      this.uploadFile(rawFile)
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
      const rawFile = files[0]
      if (!rawFile) return
      this.uploadFile(rawFile)
    },

    // SHA256提交处理
    handleSha256Submit() {
      if (!this.isValidSha256) {
        this.$message.error('请输入有效的64位SHA256哈希值')
        return
      }

      const sha256 = this.sha256Input.trim().toLowerCase()
      this.detectBySha256(sha256)
    },

    async uploadFile(rawFile) {
      if (!this.apiBaseUrl) {
        this.$message.error('API地址未加载完成，请稍后重试')
        return
      }

      this.uploadResult = null
      this.behaviour_results = {}
      this.results = []
      this.loading = true

      const formData = new FormData()
      formData.append('file', rawFile)

      try {
        console.log('开始上传文件到:', `${this.apiBaseUrl}/upload`)
        const response = await apiService.post(`${this.apiBaseUrl}/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 600000 // 单独设置上传超时（10分钟）
        })

        console.log('上传响应状态:', response.status)
        console.log('上传响应数据:', response.data)
        this.uploadResult = response.data

        // 统一化SHA256字段
        this.normalizeSha256Fields()

        // 上传成功后获取详细信息
        await this.fetchDetailAPI()
      } catch (error) {
        console.error('文件上传失败:', error)
        let errMsg = '文件上传失败！'
        if (error.code === 'ECONNABORTED') {
          errMsg = '文件上传超时（已设置10分钟，请检查后端处理速度）'
        } else if (error.response?.status === 401) {
          errMsg = '登录状态失效，请重新登录'
        } else if (error.response?.data?.message) {
          errMsg = error.response.data.message
        } else if (error.message) {
          errMsg += ' ' + error.message
        }
        this.$message.error(errMsg)
      } finally {
        this.loading = false
      }
    },

    async detectBySha256(sha256) {
      if (!this.apiBaseUrl) {
        this.$message.error('API地址未加载完成，请稍后重试')
        return
      }

      this.uploadResult = null
      this.behaviour_results = {}
      this.results = []
      this.loading = true

      try {
        console.log('开始SHA256检测:', `${this.apiBaseUrl}/detect_by_sha256`)
        const response = await apiService.post(`${this.apiBaseUrl}/detect_by_sha256`, {
          sha256: sha256
        }, {
          timeout: 600000
        })

        console.log('SHA256检测响应状态:', response.status)
        console.log('SHA256检测响应数据:', response.data)

        if (response.data && response.data.query_result) {
          this.uploadResult = response.data

          // 统一化SHA256字段
          this.normalizeSha256Fields()

          // 获取详细信息
          await this.fetchDetailAPI()
        } else {
          throw new Error('返回数据格式不正确')
        }
      } catch (error) {
        console.error('SHA256检测失败:', error)
        let errMsg = 'SHA256检测失败！'
        if (error.code === 'ECONNABORTED') {
          errMsg = '检测请求超时（已设置10分钟）'
        } else if (error.response?.status === 401) {
          errMsg = '登录状态失效，请重新登录'
        } else if (error.response?.status === 404) {
          errMsg = '未找到该SHA256对应的文件信息'
        } else if (error.response?.data?.message) {
          errMsg = error.response.data.message
        } else if (error.message) {
          errMsg += ' ' + error.message
        }
        this.$message.error(errMsg)
      } finally {
        this.loading = false
      }
    },

    // 统一化SHA256字段
    normalizeSha256Fields() {
      if (!this.uploadResult) return

      console.log('normalizeSha256Fields: 原始uploadResult:', this.uploadResult)

      // 提取query_result用于后续使用
      const queryResult = this.uploadResult.query_result || {}

      // 获取可能的SHA256值（按优先级）
      const possibleSha256Values = [
        // 最直接的sha256字段
        this.uploadResult.sha256,
        // query_result中的各种可能键名
        queryResult['SHA-256'],
        queryResult.SHA256,
        queryResult['SHA256'],
        // 其他可能的位置
        this.uploadResult.VT_API,
        this.uploadResult.VT_API
      ]

      // 找到第一个有效的SHA256值
      let sha256 = ''
      for (const value of possibleSha256Values) {
        if (value && typeof value === 'string' && value.length === 64) {
          sha256 = value.toLowerCase()
          break
        }
      }

      // 如果没有找到有效的SHA256，尝试从原始数据中查找
      if (!sha256) {
        // 深度搜索可能的SHA256字段
        const searchInObject = (obj, depth = 0) => {
          if (depth > 3) return null // 防止无限递归

          for (const key in obj) {
            if (obj[key] && typeof obj[key] === 'string' && obj[key].length === 64) {
              // 简单的SHA256格式验证（64位十六进制）
              if (/^[a-fA-F0-9]{64}$/.test(obj[key])) {
                return obj[key].toLowerCase()
              }
            } else if (obj[key] && typeof obj[key] === 'object') {
              const result = searchInObject(obj[key], depth + 1)
              if (result) return result
            }
          }
          return null
        }

        sha256 = searchInObject(this.uploadResult) || ''
      }

      console.log('normalizeSha256Fields: 统一后的SHA256:', sha256)

      // 统一设置字段
      this.uploadResult.sha256 = sha256
      if (this.uploadResult.query_result) {
        this.uploadResult.query_result.SHA256 = sha256
      }
      if (!this.uploadResult.VT_API) {
        this.uploadResult.VT_API = sha256
      }
    },

    async fetchDetailAPI() {
      if (!this.apiBaseUrl) {
        this.$message.error('API地址未加载完成')
        return
      }

      console.log('开始获取详细信息...')
      console.log('uploadResult 完整结构:', JSON.stringify(this.uploadResult, null, 2))

      this.isElLoading = true
      this.isElLoadings = true
      this.results = []
      this.behaviour_results = {}
      this.isError = false
      this.isErrors = false

      // 检查是否有必要的信息
      if (!this.uploadResult || !this.uploadResult.sha256) {
        console.warn('缺少文件SHA256信息，无法获取检测详情')
        console.warn('uploadResult:', this.uploadResult)
        this.isElLoading = false
        this.isElLoadings = false
        this.$message.warning('缺少文件SHA256信息，无法获取检测详情')
        return
      }

      const sha256 = this.uploadResult.sha256
      const VT_API = this.uploadResult.VT_API || sha256

      console.log('获取详细信息参数:', { sha256, VT_API })
      console.log('API地址:', this.apiBaseUrl)

      // 并行发起两个请求
      const detectionPromise = this.fetchDetectionData(sha256, VT_API)
      const behaviourPromise = this.fetchBehaviourData(sha256, VT_API)

      // 等待两个请求都完成
      await Promise.all([detectionPromise, behaviourPromise])

      console.log('所有详细信息获取完成')
    },

    async fetchDetectionData(sha256, VT_API) {
      try {
        console.log(`开始获取杀软检测数据: ${this.apiBaseUrl}/detection_API/${sha256}`)
        console.log(`参数: sha256=${sha256}, VT_API=${VT_API}`)
        console.log(`实际请求URL: ${this.apiBaseUrl}/detection_API/${sha256}?VT_API=${encodeURIComponent(VT_API)}`)

        const response = await apiService.get(`${this.apiBaseUrl}/detection_API/${sha256}`, {
          params: { VT_API },
          timeout: 600000
        })

        console.log('杀软检测响应状态:', response.status)
        console.log('杀软检测响应数据:', response.data)

        if (Array.isArray(response.data) && response.data.length > 0) {
          this.results = response.data
          console.log(`成功获取 ${response.data.length} 条杀软检测结果`)
        } else {
          console.warn('杀软检测数据格式异常:', response.data)
          this.isErrors = true
          this.$message.warning('杀软检测数据格式异常或为空')
        }
      } catch (error) {
        console.error('获取杀软检测数据失败:', error)
        console.error('错误详情:', error.response?.data)
        this.isErrors = true

        let errorMsg = '获取杀软检测数据失败'
        if (error.response?.status === 400) {
          errorMsg = 'SHA256格式错误'
        } else if (error.response?.status === 401) {
          errorMsg = '登录状态失效，请重新登录'
        } else if (error.response?.status === 404) {
          errorMsg = '杀软检测API接口不存在或样本文件不存在'
        } else if (error.response?.status === 500) {
          errorMsg = '杀软检测服务器错误'
        } else if (error.code === 'ECONNABORTED') {
          errorMsg = '杀软检测请求超时'
        } else if (error.message) {
          errorMsg += ': ' + error.message
        }

        this.$message.error(errorMsg)
      } finally {
        this.isElLoading = false
      }
    },

    async fetchBehaviourData(sha256, VT_API) {
      try {
        console.log(`开始获取动态检测数据: ${this.apiBaseUrl}/behaviour_API/${sha256}`)
        console.log(`参数: sha256=${sha256}, VT_API=${VT_API}`)
        console.log(`实际请求URL: ${this.apiBaseUrl}/behaviour_API/${sha256}?VT_API=${encodeURIComponent(VT_API)}`)

        const response = await apiService.get(`${this.apiBaseUrl}/behaviour_API/${sha256}`, {
          params: { VT_API },
          timeout: 600000
        })

        console.log('动态检测响应状态:', response.status)
        console.log('动态检测响应数据:', response.data)

        if (response.data && typeof response.data === 'object') {
          // 检查是否有错误消息
          if (response.data.message) {
            console.warn('动态检测API返回消息:', response.data.message)
            this.isError = true

            // 如果是空数据消息，不显示错误，只记录日志
            if (response.data.message.includes('无动态行为数据') ||
                response.data.message.includes('no dynamic behaviour data')) {
              console.log('动态检测无数据')
              this.behaviour_results = {}
            } else {
              this.$message.warning(response.data.message)
            }
          } else {
            // 正常数据
            this.behaviour_results = response.data
            console.log('动态检测数据已设置，hasValidData:', this.hasValidData)

            // 如果没有有效数据，设置错误状态
            if (!this.hasValidData) {
              this.isError = true
            }
          }
        } else {
          console.warn('动态检测数据格式异常:', response.data)
          this.isError = true
          this.$message.warning('动态检测数据格式异常')
        }
      } catch (error) {
        console.error('获取动态检测数据失败:', error)
        console.error('错误详情:', error.response?.data)
        this.isError = true

        let errorMsg = '获取动态检测数据失败'
        if (error.response?.status === 400) {
          errorMsg = 'SHA256格式错误'
        } else if (error.response?.status === 401) {
          errorMsg = '登录状态失效，请重新登录'
        } else if (error.response?.status === 404) {
          errorMsg = '动态检测API接口不存在或样本文件不存在'
        } else if (error.response?.status === 500) {
          errorMsg = '动态检测服务器错误'
        } else if (error.code === 'ECONNABORTED') {
          errorMsg = '动态检测请求超时'
        } else if (error.message) {
          errorMsg += ': ' + error.message
        }

        this.$message.error(errorMsg)
      } finally {
        this.isElLoadings = false
        console.log('动态检测加载完成，isError:', this.isError, 'hasValidData:', this.hasValidData)
      }
    }
  }
}
</script>

<style scoped>
/* 基础样式 */
.file-upload-input{
  display: none;
  z-index: -9999;
}

/* 检测模式切换容器 */
.detection-mode-container {
  text-align: center;
  margin: 20px 0;
}

/* 检测内容区域 */
.detection-content {
  margin: 20px auto;
  width: 80%;
  max-width: 800px;
}

/* 文件上传区域 */
.drop{
  border: 2px dashed #bbb;
  width: 100%;
  height: 160px;
  line-height: 160px;
  margin: 0 auto;
  font-size: 24px;
  border-radius: 5px;
  text-align: center;
  color: #bbb;
  position: relative;
}

/* SHA256输入区域 */
.sha256-input-container {
  margin-top: 20px;
}

.sha256-input {
  font-family: 'Courier New', monospace;
  font-size: 16px;
}

.sha256-input :deep(.el-input-group__prepend) {
  padding: 0 12px;
  background-color: #f5f7fa;
}

.sha256-input :deep(.el-input-group__append) {
  padding: 0;
  width: 180px;
}

/* 按钮组样式 */
.button-group {
  display: flex;
  gap: 8px;
}

.button-group .el-button {
  flex: 1;
  min-width: 70px;
}

/* 高亮按钮样式 */
.highlight-button {
  background-color: #409eff !important;
  border-color: #409eff !important;
  color: white !important;
}

.highlight-button:hover {
  background-color: #66b1ff !important;
  border-color: #66b1ff !important;
}

/* 输入验证提示 */
.input-validation {
  margin-top: 10px;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 4px;
  border: 1px solid #e9ecef;
}

.char-count {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.char-count-valid {
  color: #67c23a;
  font-weight: bold;
}

.char-count-exceed {
  color: #f56c6c;
  font-weight: bold;
}

.char-count-normal {
  color: #909399;
}

.validation-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  padding: 6px 0;
}

.validation-hint.success {
  color: #67c23a;
}

.input-hint {
  margin-top: 20px;
}

.input-hint :deep(.el-alert) {
  padding: 12px 16px;
}

.input-hint code {
  display: block;
  margin-top: 8px;
  padding: 8px;
  background-color: #f5f5f5;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  overflow-x: auto;
}

.hint-tips {
  margin-top: 8px;
  font-size: 13px;
  color: #909399;
}

/* 检测模式标签 */
.detection-mode-tag {
  margin-top: 10px;
  text-align: center;
  display: flex;
  justify-content: center;
  align-items: center;
}

/* 结果状态 */
.result-status {
  font-size: 30px;
  color: #333;
  font-weight: bold;
  text-align: center;
  display: inline-block;
  margin-top: 20px;
}

/* 导航按钮容器 */
.custom-button {
  margin-top: 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: calc(25% - 10px);
  height: 50px;
  background-color: white;
  border: 1px solid #ccc;
  cursor: pointer;
  padding: 0 10px;
  font-size: 16px;
  color: #333;
  transition: background-color 0.3s ease;
}

.custom-button:hover {
  background-color: #f5f5f5;
}

.active-button {
  background-color: #409EFF;
  color: white;
  border-color: #409EFF;
}

.active-button:hover {
  background-color: #66b1ff;
}

.button-icon {
  width: 24px;
  height: 24px;
  margin-right: 8px;
  margin-top: 2px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .detection-content {
    width: 95%;
  }

  .drop {
    width: 100%;
    font-size: 18px;
    height: 120px;
    line-height: 120px;
  }

  .sha256-input {
    font-size: 14px;
  }

  .sha256-input :deep(.el-input-group__append) {
    width: 160px;
  }

  .button-group .el-button {
    min-width: 60px;
    font-size: 12px;
    padding: 8px 6px;
  }

  .custom-button {
    width: calc(25% - 5px);
    font-size: 14px;
    padding: 0 5px;
  }

  .button-icon {
    width: 18px;
    height: 18px;
    margin-right: 4px;
  }

  .result-status {
    font-size: 24px;
  }
}

@media (max-width: 480px) {
  .drop {
    font-size: 16px;
    height: 100px;
    line-height: 100px;
  }

  .custom-button {
    flex-direction: column;
    height: 60px;
    padding: 5px;
  }

  .button-icon {
    margin-right: 0;
    margin-bottom: 2px;
  }

  .detection-mode-tag {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
  }

  .detection-mode-tag .el-tag {
    width: fit-content;
  }

  .sha256-input :deep(.el-input-group__append) {
    width: 140px;
  }

  .button-group {
    flex-direction: column;
    gap: 4px;
  }

  .button-group .el-button {
    min-width: auto;
    padding: 4px 8px;
  }
}
</style>
