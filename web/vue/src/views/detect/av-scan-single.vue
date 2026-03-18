<template>
  <div class="av-scan-single-container">
    <!-- 标题 -->
    <div class="text-center">
      <h2 class="text-primary">分布式杀毒软件检测系统</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
    </div>

    <!-- 文件上传区域 -->
    <div class="upload-section">
      <input
        ref="file-upload-input"
        class="file-upload-input"
        type="file"
        @change="handleFileChange"
      >
      <div class="drop-zone" @drop="handleDrop" @dragover="handleDragover">
        <!-- 初始状态:选择文件 -->
        <div v-if="!uploading && !scanning && !scanResult">
          <svg-icon icon-class="upload" class="upload-icon" />
          <p class="drop-text">把待检文件拖到这里或</p>
          <el-button
            type="primary"
            size="large"
            @click="handleUpload"
          >
            选择待检文件
          </el-button>
        </div>

        <!-- 检测中状态 -->
        <div v-else-if="uploading || scanning" class="uploading-state">
          <i class="el-icon-loading" style="font-size: 48px; color: #409EFF;"></i>
          <p class="uploading-text">{{ scanning ? '正在检测中...' : '正在上传文件...' }}</p>
          <p v-if="selectedFile" class="file-info">{{ selectedFile.name }} ({{ formatFileSize(selectedFile.size) }})</p>
        </div>

        <!-- 检测完成,显示结果 -->
        <div v-else-if="scanResult" class="result-preview">
          <svg-icon icon-class="detect-report" class="result-icon" />
          <p class="result-text">检测完成</p>
          <el-button type="primary" @click="resetScan">继续检测</el-button>
        </div>
      </div>
    </div>

    <!-- 扫描结果区域 -->
    <div v-if="scanResult" class="result-section">
      <!-- 结果统计 -->
      <div class="result-summary">
        <div class="summary-header">
          <svg-icon icon-class="detect-report" class="report-icon" />
          <span class="summary-title">检测结果</span>
        </div>
        <div class="summary-stats">
          <div class="stat-item">
            <span class="stat-label">文件名:</span>
            <span class="stat-value">{{ scanResult.file_name }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">文件大小:</span>
            <span class="stat-value">{{ scanResult.file_size }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">扫描时间:</span>
            <span class="stat-value">{{ scanResult.elapsed_seconds }}秒</span>
          </div>
          <div class="stat-item highlight">
            <span class="stat-label">检测结果:</span>
            <span class="stat-value" :class="resultClass">
              {{ scanResult.total_engines }}个杀软中 {{ scanResult.malicious_count }}个判定为恶意
            </span>
          </div>
        </div>
      </div>

      <!-- 引擎检测结果 -->
      <div class="engines-result">
        <h3 class="engines-title">引擎检测结果</h3>
        <div class="engines-grid">
          <div
            v-for="engine in scanResult.engines"
            :key="engine.name"
            class="engine-card"
            :class="getEngineClass(engine.status)"
          >
            <div class="engine-header">
              <img
                :src="getEngineIcon(engine.name)"
                :alt="engine.name"
                class="engine-icon"
                @error="handleIconError"
              >
              <span class="engine-name">{{ engine.name }}</span>
            </div>
            <div class="engine-status">
              <span class="status-icon">{{ getStatusIcon(engine.status) }}</span>
              <span class="status-text">{{ getStatusText(engine.status) }}</span>
            </div>
            <div class="engine-info">
              <span class="engine-vm">VM: {{ engine.vm }}</span>
              <span class="engine-time">{{ engine.elapsed_seconds }}s</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 操作按钮 -->
      <div class="action-buttons">
        <el-button type="primary" @click="resetScan">继续检测</el-button>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

// 创建axios实例
const apiService = axios.create({
  timeout: 600000, // 10分钟超时
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
  name: 'AVScanSingle',
  data() {
    return {
      selectedFile: null,
      uploading: false,
      scanning: false,
      scanResult: null,
      apiBaseUrl: '', // 初始为空,等待从配置文件加载
      avEngines: [
        'Avira', 'McAfee', 'WindowsDefender', 'IkarusT3', 'Emsisoft',
        'FProtect', 'Vba32', 'ClamAV', 'Kaspersky', 'ESET',
        'DrWeb', 'Avast', 'AVG', 'AdAware', 'FSecure'
      ]
    }
  },
  computed: {
    resultClass() {
      if (!this.scanResult) return ''
      return this.scanResult.malicious_count > 0 ? 'malicious' : 'safe'
    }
  },
  created() {
    console.log('=== av-scan-single 组件创建 ===')
    this.loadConfig()
  },
  mounted() {
    console.log('=== av-scan-single 组件挂载 ===')
    console.log('file-upload-input ref:', this.$refs['file-upload-input'])
    console.log('API地址:', this.apiBaseUrl)
  },
  methods: {
    // 加载配置文件
    async loadConfig() {
      try {
        const response = await apiService.get('/config.ini', {
          responseType: 'text',
          timeout: 5000
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
      } catch (error) {
        console.warn('加载配置文件失败:', error.message)
        // 使用兜底地址
        this.apiBaseUrl = 'http://192.168.8.202:5005'
        console.log('使用兜底API地址:', this.apiBaseUrl)
      }

      // 验证API地址是否加载成功
      if (!this.apiBaseUrl) {
        console.error('API地址未设置!')
        this.$message.error('API地址配置失败,请联系管理员')
      } else {
        console.log('最终API地址:', this.apiBaseUrl)
      }
    },

    // 文件拖拽处理
    handleDrop(e) {
      e.stopPropagation()
      e.preventDefault()
      if (this.uploading || this.scanning) return

      const files = e.dataTransfer.files
      if (files.length !== 1) {
        this.$message.error('只支持上传一个文件!')
        return
      }
      this.selectedFile = files[0]
      console.log('拖拽文件:', this.selectedFile.name)
      this.$message.success(`已选择文件: ${this.selectedFile.name}`)

      // 自动开始检测
      console.log('自动开始检测...')
      this.$nextTick(() => {
        this.startScan()
      })
    },

    handleDragover(e) {
      e.stopPropagation()
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    },

    handleUpload() {
      console.log('=== handleUpload 被调用 ===')
      console.log('file-upload-input ref:', this.$refs['file-upload-input'])

      try {
        if (this.$refs['file-upload-input']) {
          console.log('触发文件选择器点击')
          this.$refs['file-upload-input'].click()
        } else {
          console.error('错误: file-upload-input ref未找到')
          this.$message.error('文件上传组件初始化失败,请刷新页面重试')
        }
      } catch (error) {
        console.error('handleUpload执行错误:', error)
        this.$message.error('打开文件选择器失败: ' + error.message)
      }
    },

    handleFileChange(e) {
      console.log('=== handleFileChange 被调用 ===')
      console.log('事件对象:', e)
      console.log('文件列表:', e.target.files)

      const files = e.target.files
      if (files.length > 0) {
        this.selectedFile = files[0]
        console.log('已选择文件:', this.selectedFile.name, '大小:', this.selectedFile.size)
        this.$message.success(`已选择文件: ${this.selectedFile.name}`)

        // 自动开始检测
        console.log('自动开始检测...')
        this.$nextTick(() => {
          this.startScan()
        })
      } else {
        console.log('未选择任何文件')
      }
    },

    // 开始扫描
    async startScan() {
      console.log('=== startScan 被调用 ===')
      console.log('selectedFile:', this.selectedFile)

      if (!this.selectedFile) {
        this.$message.error('请先选择文件')
        return
      }

      console.log('API地址:', this.apiBaseUrl)
      console.log('文件信息:', {
        name: this.selectedFile.name,
        size: this.selectedFile.size,
        type: this.selectedFile.type
      })

      this.scanning = true
      this.uploading = true

      // 初始化引擎结果列表(15个引擎,初始状态都是检测中)
      this.scanResult = {
        file_name: this.selectedFile.name,
        file_size: this.formatFileSize(this.selectedFile.size),
        total_engines: 15,
        malicious_count: 0,
        safe_count: 0,
        engines: this.avEngines.map(engine => ({
          name: engine,
          status: 'scanning', // scanning/safe/malicious/unsupported/error
          vm: 'unknown',
          elapsed_seconds: 0
        }))
      }

      const formData = new FormData()
      formData.append('file', this.selectedFile)

      console.log('准备发送流式API请求到:', `${this.apiBaseUrl}/api/av_scan_single_streaming`)

      try {
        // 使用fetch API支持流式响应
        const response = await fetch(`${this.apiBaseUrl}/api/av_scan_single_streaming`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token') || sessionStorage.getItem('token')}`
          },
          body: formData
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = '' // 缓冲区,用于处理不完整的数据

        // 读取流式数据
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          // 将新数据添加到缓冲区
          buffer += decoder.decode(value, { stream: true })

          // 按行分割数据
          const lines = buffer.split('\n')

          // 保留最后一行(可能不完整)
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const jsonStr = line.substring(6).trim()
                if (!jsonStr) continue // 跳过空行

                const data = JSON.parse(jsonStr)
                console.log('收到流式数据:', data)

                // 处理不同类型的数据
                if (data.type === 'file_info') {
                  // 文件信息
                  this.scanResult.file_name = data.file_name
                  this.scanResult.file_size = data.file_size
                } else if (data.type === 'complete') {
                  // 扫描完成
                  this.scanResult.elapsed_seconds = data.elapsed_seconds
                  this.scanResult.total_engines = data.total_engines
                  this.$message.success('扫描完成!')
                } else if (data.engine) {
                  // 引擎结果 - 立即更新对应的引擎
                  this.updateEngineResult(data)
                }
              } catch (e) {
                console.error('解析数据失败:', e, '原始数据:', line)
              }
            }
          }
        }

      } catch (error) {
        console.error('扫描失败:', error)
        let errMsg = '扫描失败!'
        if (error.message) {
          errMsg += ' ' + error.message
        }
        this.$message.error(errMsg)
      } finally {
        this.scanning = false
        this.uploading = false
      }
    },

    // 更新单个引擎结果
    updateEngineResult(data) {
      if (!this.scanResult || !this.scanResult.engines) return

      // 找到对应的引擎并更新
      const engineIndex = this.scanResult.engines.findIndex(e => e.name === data.engine)
      if (engineIndex !== -1) {
        this.scanResult.engines[engineIndex] = {
          name: data.engine,
          status: data.status,
          vm: data.vm_id || 'unknown',
          elapsed_seconds: data.elapsed_seconds || 0,
          error: data.error
        }

        // 更新统计
        this.updateStatistics()
      }
    },

    // 更新统计信息
    updateStatistics() {
      if (!this.scanResult || !this.scanResult.engines) return

      let maliciousCount = 0
      let safeCount = 0

      this.scanResult.engines.forEach(engine => {
        if (engine.status === 'malicious') {
          maliciousCount++
        } else if (engine.status === 'safe') {
          safeCount++
        }
      })

      this.scanResult.malicious_count = maliciousCount
      this.scanResult.safe_count = safeCount
    },

    // 重置扫描
    resetScan() {
      this.selectedFile = null
      this.scanResult = null
      this.scanning = false
      this.uploading = false
      // 清空文件输入
      if (this.$refs['file-upload-input']) {
        this.$refs['file-upload-input'].value = ''
      }
    },

    // 格式化文件大小
    formatFileSize(bytes) {
      if (bytes === 0) return '0 B'
      const k = 1024
      const sizes = ['B', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
    },

    // 获取引擎图标
    getEngineIcon(engineName) {
      // 直接返回图标路径,Vite会自动处理
      try {
        // 使用动态导入
        return new URL(`../../assets/antivirus-icons/${engineName}.png`, import.meta.url).href
      } catch (e) {
        // 如果图标不存在,返回空字符串
        console.warn(`图标不存在: ${engineName}`)
        return ''
      }
    },

    // 图标加载失败处理
    handleIconError(e) {
      // 图标加载失败时,隐藏图标
      e.target.style.display = 'none'
    },

    // 获取引擎状态样式类
    getEngineClass(status) {
      const classMap = {
        'scanning': 'engine-scanning',
        'safe': 'engine-safe',
        'malicious': 'engine-malicious',
        'unsupported': 'engine-unsupported',
        'error': 'engine-error'
      }
      return classMap[status] || ''
    },

    // 获取状态图标
    getStatusIcon(status) {
      const iconMap = {
        'scanning': '⏳',
        'safe': '🟢',
        'malicious': '🔴',
        'unsupported': '⚪',
        'error': '⚠️'
      }
      return iconMap[status] || '❓'
    },

    // 获取状态文本
    getStatusText(status) {
      const textMap = {
        'scanning': '检测中...',
        'safe': '安全',
        'malicious': '恶意',
        'unsupported': '不支持',
        'error': '错误'
      }
      return textMap[status] || '未知'
    }
  }
}
</script>

<style scoped>
.av-scan-single-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

/* 上传区域 */
.upload-section {
  margin: 30px auto;
  max-width: 800px;
}

.file-upload-input {
  display: none;
}

.drop-zone {
  border: 2px dashed #dcdfe6;
  border-radius: 8px;
  padding: 60px 20px;
  text-align: center;
  background-color: #fafafa;
  transition: all 0.3s;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.drop-zone:hover {
  border-color: #409EFF;
  background-color: #f0f7ff;
}

.upload-icon {
  width: 80px;
  height: 80px;
  color: #909399;
  margin-bottom: 20px;
}

.drop-text {
  font-size: 18px;
  color: #606266;
  margin-bottom: 20px;
}

/* 上传中状态 */
.uploading-state {
  text-align: center;
}

.uploading-text {
  margin-top: 20px;
  font-size: 18px;
  color: #409EFF;
  font-weight: bold;
}

.file-info {
  margin-top: 10px;
  font-size: 14px;
  color: #909399;
}

/* 结果预览 */
.result-preview {
  text-align: center;
}

.result-icon {
  width: 60px;
  height: 60px;
  color: #67c23a;
  margin-bottom: 15px;
}

.result-text {
  font-size: 18px;
  color: #67c23a;
  font-weight: bold;
  margin-bottom: 20px;
}

/* 文件已选择 */
.file-selected {
  text-align: center;
}

.file-icon {
  width: 60px;
  height: 60px;
  color: #409EFF;
  margin-bottom: 15px;
}

.file-name {
  font-size: 18px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 10px;
}

.file-size {
  font-size: 14px;
  color: #909399;
  margin-bottom: 20px;
}

.button-group {
  display: flex;
  gap: 10px;
  justify-content: center;
}

/* 结果区域 */
.result-section {
  margin-top: 40px;
}

.result-summary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 30px;
  border-radius: 12px;
  margin-bottom: 30px;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.summary-header {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.report-icon {
  width: 32px;
  height: 32px;
  margin-right: 12px;
}

.summary-title {
  font-size: 24px;
  font-weight: bold;
}

.summary-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.stat-item {
  display: flex;
  flex-direction: column;
}

.stat-label {
  font-size: 14px;
  opacity: 0.9;
  margin-bottom: 5px;
}

.stat-value {
  font-size: 18px;
  font-weight: bold;
}

.stat-item.highlight .stat-value {
  font-size: 20px;
  padding: 8px 12px;
  background-color: rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  display: inline-block;
}

.stat-value.malicious {
  color: #f56c6c;
}

.stat-value.safe {
  color: #67c23a;
}

/* 引擎结果 */
.engines-result {
  background: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.engines-title {
  font-size: 20px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 25px;
  padding-bottom: 15px;
  border-bottom: 2px solid #f0f0f0;
}

.engines-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}

.engine-card {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 20px;
  transition: all 0.3s;
  border: 2px solid transparent;
}

.engine-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.engine-card.engine-scanning {
  border-color: #409EFF;
  background: linear-gradient(135deg, #f0f7ff 0%, #e6f1ff 100%);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

.engine-card.engine-safe {
  border-color: #67c23a;
  background: linear-gradient(135deg, #f0f9ff 0%, #e1f3d8 100%);
}

.engine-card.engine-malicious {
  border-color: #f56c6c;
  background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
}

.engine-card.engine-unsupported {
  border-color: #909399;
  background: #f5f5f5;
}

.engine-card.engine-error {
  border-color: #e6a23c;
  background: linear-gradient(135deg, #fffbf0 0%, #fff3e0 100%);
}

.engine-header {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.engine-icon {
  width: 32px;
  height: 32px;
  min-width: 32px;
  min-height: 32px;
  max-width: 32px;
  max-height: 32px;
  margin-right: 12px;
  border-radius: 4px;
  object-fit: contain;
  background-color: #f5f5f5;
  padding: 2px;
}

.engine-name {
  font-size: 16px;
  font-weight: bold;
  color: #303133;
}

.engine-status {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.status-icon {
  font-size: 20px;
  margin-right: 8px;
}

.status-text {
  font-size: 16px;
  font-weight: bold;
}

.engine-info {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #909399;
}

.engine-vm,
.engine-time {
  display: inline-block;
}

/* 操作按钮 */
.action-buttons {
  text-align: center;
  margin-top: 30px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .engines-grid {
    grid-template-columns: 1fr;
  }

  .summary-stats {
    grid-template-columns: 1fr;
  }

  .drop-zone {
    padding: 40px 15px;
    min-height: 200px;
  }

  .upload-icon {
    width: 60px;
    height: 60px;
  }

  .drop-text {
    font-size: 16px;
  }
}
</style>
