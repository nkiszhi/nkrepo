<template>
  <div class="av-scan-batch-v2-container">
    <!-- 标题 -->
    <div class="text-center">
      <h2 class="text-primary">批量杀毒软件检测</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG - 支持引擎选择</p>
    </div>

    <!-- 步骤指示器 -->
    <el-steps :active="currentStep" finish-status="success" simple class="steps-container">
      <el-step title="上传文件" icon="el-icon-upload"></el-step>
      <el-step title="选择引擎" icon="el-icon-setting"></el-step>
      <el-step title="检测进度" icon="el-icon-video-play"></el-step>
      <el-step title="检测结果" icon="el-icon-document"></el-step>
    </el-steps>

    <!-- 第一步:上传文件 -->
    <div v-show="currentStep === 0" class="step-content">
      <div class="upload-section">
        <input
          ref="file-upload-input"
          class="file-upload-input"
          type="file"
          multiple
          @change="handleFileChange"
        >
        <div class="drop-zone" @drop="handleDrop" @dragover="handleDragover">
          <div v-if="!uploading">
            <svg-icon icon-class="upload" class="upload-icon" />
            <p class="drop-text">把待检文件拖到这里或</p>
            <el-button type="primary" size="large" @click="handleUpload">
              选择待检文件(支持多选)
            </el-button>
          </div>
          <div v-else class="uploading-state">
            <i class="el-icon-loading" style="font-size: 48px; color: #409EFF;"></i>
            <p class="uploading-text">正在上传文件...</p>
          </div>
        </div>
      </div>

      <!-- 引擎列表显示 -->
      <div class="engine-info-section">
        <div class="engine-info-header">
          <h3>支持检测引擎列表</h3>
          <span class="engine-count">共 {{ engineList.length }} 个引擎</span>
        </div>
        <el-table :data="engineList" border style="width: 100%;" size="small">
          <el-table-column type="index" label="序号" width="60" align="center" />
          <el-table-column prop="name" label="引擎名称" align="center" />
          <el-table-column prop="version" label="版本号" align="center" />
        </el-table>
      </div>

      <!-- 已上传文件列表 -->
      <div v-if="uploadedFiles.length > 0" class="files-list-section">
        <div class="list-header">
          <h3>已上传文件列表</h3>
          <div class="list-actions">
            <span class="file-count">共上传: {{ uploadedFiles.length }}个文件</span>
            <el-button type="danger" size="small" @click="clearAllFiles">清空所有</el-button>
          </div>
        </div>
        <el-table :data="uploadedFiles" border style="width: 100%">
          <el-table-column type="index" label="序号" width="60"></el-table-column>
          <el-table-column prop="name" label="文件名" min-width="200"></el-table-column>
          <el-table-column prop="size" label="文件大小" width="120"></el-table-column>
          <el-table-column label="操作" width="100">
            <template #default="scope">
              <el-button type="text" size="small" @click="removeFile(scope.$index)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>

        <div class="next-step-button">
          <el-button type="primary" size="large" @click="goToEngineSelection">
            下一步：选择引擎
          </el-button>
        </div>
      </div>
    </div>

    <!-- 第二步:选择引擎 -->
    <div v-show="currentStep === 1" class="step-content">
      <div class="engine-selection-section">
        <div class="selection-header">
          <el-checkbox v-model="selectAllEngines" @change="handleSelectAll">
            全选
          </el-checkbox>
          <span class="selected-count">已选择 {{ selectedEngines.length }} / {{ allEngines.length }} 个引擎</span>
        </div>

        <div class="engines-grid">
          <div
            v-for="engine in allEngines"
            :key="engine"
            class="engine-card"
            :class="{ 'selected': selectedEngines.includes(engine) }"
            @click="handleEngineSelect(engine)"
          >
            <img :src="getEngineIcon(engine)" class="engine-icon" @error="handleIconError" />
            <span class="engine-name">{{ engine }}</span>
          </div>
        </div>

        <div class="action-buttons">
          <el-button @click="currentStep = 0">上一步</el-button>
          <el-button type="primary" @click="startBatchScan" :disabled="selectedEngines.length === 0">
            开始检测
          </el-button>
        </div>
      </div>
    </div>

    <!-- 第三步:检测进度 -->
    <div v-show="currentStep === 2" class="step-content">
      <div class="progress-section">
        <div class="progress-header">
          <h3>{{ progressTitle }}</h3>
          <el-tag :type="getTaskStatusType(taskStatus)" size="large">
            {{ getTaskStatusText(taskStatus) }}
          </el-tag>
        </div>

        <el-progress
          :percentage="progress"
          :status="progressStatus"
          :stroke-width="25"
          class="progress-bar"
        />

        <div class="progress-detail">
          <div class="detail-card">
            <div class="card-icon"><i class="el-icon-document"></i></div>
            <div class="card-content">
              <div class="card-label">文件进度</div>
              <div class="card-value">{{ scannedFiles }} / {{ totalFiles }}</div>
            </div>
          </div>
          <div class="detail-card">
            <div class="card-icon"><i class="el-icon-time"></i></div>
            <div class="card-content">
              <div class="card-label">已用时间</div>
              <div class="card-value">{{ formatTime(elapsedSeconds) }}</div>
              <div class="card-desc">预计剩余: {{ formatTime(estimatedRemaining) }}</div>
            </div>
          </div>
          <div class="detail-card" v-if="currentFileName">
            <div class="card-icon"><i class="el-icon-loading"></i></div>
            <div class="card-content">
              <div class="card-label">当前文件</div>
              <div class="card-value file-name-text">{{ currentFileName }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 第四步:检测结果 -->
    <div v-show="currentStep === 3" class="step-content">
      <div class="result-section">
        <div class="result-header">
          <h3>检测结果</h3>
          <div class="result-actions">
            <el-tag type="success" size="large">检测完成</el-tag>
            <el-button type="success" @click="downloadReport" :loading="downloading">
              <svg-icon icon-class="download" style="margin-right: 8px;" />
              下载CSV报告
            </el-button>
          </div>
        </div>

        <!-- 统计信息 -->
        <div class="result-summary">
          <div class="summary-item">
            <div class="summary-icon" style="background: #67c23a;"><i class="el-icon-check"></i></div>
            <div class="summary-content">
              <div class="summary-value">{{ safeFilesCount }}</div>
              <div class="summary-label">安全文件</div>
            </div>
          </div>
          <div class="summary-item">
            <div class="summary-icon" style="background: #f56c6c;"><i class="el-icon-warning"></i></div>
            <div class="summary-content">
              <div class="summary-value">{{ maliciousFilesCount }}</div>
              <div class="summary-label">恶意文件</div>
            </div>
          </div>
          <div class="summary-item">
            <div class="summary-icon" style="background: #409EFF;"><i class="el-icon-document"></i></div>
            <div class="summary-content">
              <div class="summary-value">{{ totalFiles }}</div>
              <div class="summary-label">总文件数</div>
            </div>
          </div>
        </div>

        <!-- 结果表格 -->
        <div class="result-table-container">
          <el-table :data="scanResults" border style="width: 100%" max-height="600">
            <el-table-column prop="file_name" label="文件名" fixed width="200"></el-table-column>
            <el-table-column
              v-for="engine in selectedEngines"
              :key="engine"
              :prop="`engines.${engine}`"
              :label="engine"
              width="100"
            >
              <template #default="scope">
                <el-tag :type="getEngineStatusType(scope.row.engines[engine])" size="small">
                  {{ getEngineStatusText(scope.row.engines[engine]) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="malicious_count" label="恶意数" width="80" fixed="right">
              <template #default="scope">
                <span :class="{'malicious-count': scope.row.malicious_count > 0}">
                  {{ scope.row.malicious_count }}
                </span>
              </template>
            </el-table-column>
          </el-table>
        </div>

        <div class="result-actions">
          <el-button type="primary" @click="resetBatchScan">继续检测</el-button>
          <el-button @click="goToHistory">查看历史记录</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

const apiService = axios.create({
  timeout: 600000,
  headers: { 'Content-Type': 'application/json' }
})

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
  name: 'AVScanBatchV2',
  data() {
    return {
      currentStep: 0,
      uploadedFiles: [],
      uploading: false,
      allEngines: [
        'Avira', 'McAfee', 'WindowsDefender', 'IkarusT3', 'Emsisoft',
        'FProtect', 'Vba32', 'ClamAV', 'Kaspersky', 'ESET',
        'DrWeb', 'Avast', 'AVG', 'AdAware', 'FSecure'
      ],
      // 引擎列表（带版本号）
      engineList: [
        { name: 'Avira', version: '1.1.115.3235' },
        { name: 'McAfee', version: '1.146' },
        { name: 'WindowsDefender', version: '1.445.6.0' },
        { name: 'IkarusT3', version: '6.04.03' },
        { name: 'Emsisoft', version: '2026.1.0.12700' },
        { name: 'FProtect', version: '6.0.9.6' },
        { name: 'Vba32', version: '3.12.18.4' },
        { name: 'ClamAV', version: '1.5.2' },
        { name: 'Kaspersky', version: '21.24.8.522' },
        { name: 'ESET', version: '19.1.12.0' },
        { name: 'DrWeb', version: '12.0.9.1140' },
        { name: 'Avast', version: '26.1.10738' },
        { name: 'AVG', version: '26.1.10738' },
        { name: 'AdAware', version: '12.3.909.11573' },
        { name: 'FSecure', version: '26.1' }
      ],
      selectedEngines: [],
      selectAllEngines: false,
      taskId: '',
      taskStatus: 'pending',
      progress: 0,
      totalFiles: 0,
      scannedFiles: 0,
      elapsedSeconds: 0,
      estimatedRemaining: 0,
      currentFileName: '',
      downloading: false,
      scanResults: [],
      safeFilesCount: 0,
      maliciousFilesCount: 0,
      apiBaseUrl: '',
      statusCheckInterval: null
    }
  },
  computed: {
    progressStatus() {
      if (this.taskStatus === 'completed') return 'success'
      if (this.taskStatus === 'failed') return 'exception'
      return null
    },
    progressTitle() {
      if (this.taskStatus === 'pending') return '准备检测...'
      if (this.taskStatus === 'running') return '检测进度'
      if (this.taskStatus === 'completed') return '检测完成'
      if (this.taskStatus === 'failed') return '检测失败'
      return '检测进度'
    }
  },
  created() {
    this.loadConfig()
  },
  beforeDestroy() {
    if (this.statusCheckInterval) {
      clearInterval(this.statusCheckInterval)
    }
  },
  methods: {
    async loadConfig() {
      try {
        const response = await apiService.get('/config.ini', { responseType: 'text', timeout: 5000 })
        const lines = response.data.split('\n')
        let inApiSection = false
        for (const line of lines) {
          const trimmedLine = line.trim()
          if (trimmedLine === '[api]') { inApiSection = true; continue }
          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=')
            if (parts.length >= 2) { this.apiBaseUrl = parts[1].trim(); break }
          }
          if (inApiSection && trimmedLine.startsWith('[')) break
        }
      } catch (error) {
        console.warn('加载配置文件失败:', error.message)
      }
    },

    handleDrop(e) {
      e.stopPropagation()
      e.preventDefault()
      if (this.uploading) return
      const files = Array.from(e.dataTransfer.files)
      this.addFiles(files)
    },

    handleDragover(e) {
      e.stopPropagation()
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    },

    handleUpload() {
      this.$refs['file-upload-input'].click()
    },

    handleFileChange(e) {
      const files = Array.from(e.target.files)
      this.addFiles(files)
      e.target.value = ''
    },

    addFiles(files) {
      files.forEach(file => {
        this.uploadedFiles.push({
          name: file.name,
          size: this.formatFileSize(file.size),
          file: file
        })
      })
      this.$message.success(`已添加 ${files.length} 个文件`)
    },

    removeFile(index) {
      this.uploadedFiles.splice(index, 1)
    },

    clearAllFiles() {
      this.uploadedFiles = []
    },

    goToEngineSelection() {
      if (this.uploadedFiles.length === 0) {
        this.$message.error('请先上传文件')
        return
      }
      this.currentStep = 1
    },

    handleSelectAll() {
      if (this.selectAllEngines) {
        this.selectedEngines = [...this.allEngines]
      } else {
        this.selectedEngines = []
      }
    },

    handleEngineSelect(engine) {
      const index = this.selectedEngines.indexOf(engine)
      if (index > -1) {
        this.selectedEngines.splice(index, 1)
      } else {
        this.selectedEngines.push(engine)
      }
      this.selectAllEngines = this.selectedEngines.length === this.allEngines.length
    },

    getEngineIcon(engine) {
      // 使用与av-scan-single.vue相同的图标路径
      try {
        return new URL(`../../assets/antivirus-icons/${engine}.png`, import.meta.url).href
      } catch (e) {
        console.warn(`图标不存在: ${engine}`)
        return ''
      }
    },

    handleIconError(e) {
      // 图标加载失败时,隐藏图标
      e.target.style.display = 'none'
    },

    async startBatchScan() {
      if (this.selectedEngines.length === 0) {
        this.$message.error('请至少选择一个引擎')
        return
      }

      this.uploading = true
      this.currentStep = 2

      try {
        // 调试日志：显示选择的引擎
        console.log('选择的引擎:', this.selectedEngines)
        console.log('引擎数量:', this.selectedEngines.length)
        const enginesStr = this.selectedEngines.join(',')
        console.log('引擎字符串:', enginesStr)
        
        const formData = new FormData()
        this.uploadedFiles.forEach(file => {
          formData.append('files', file.file)
        })
        formData.append('engines', enginesStr)
        
        // 验证 FormData 内容
        console.log('FormData engines:', formData.get('engines'))

        const uploadResponse = await apiService.post(
          `${this.apiBaseUrl}/api/av_batch_upload`,
          formData,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        )

        this.taskId = uploadResponse.data.task_id
        this.totalFiles = uploadResponse.data.total_files

        this.$message.success(`成功上传 ${this.totalFiles} 个文件，正在启动检测...`)

        this.startStatusCheck()

        await apiService.post(`${this.apiBaseUrl}/api/av_batch_scan_start`, {
          task_id: this.taskId
        })

        this.$message.success('批量检测任务已启动')

      } catch (error) {
        console.error('启动批量扫描失败:', error)
        this.$message.error('启动失败: ' + (error.response?.data?.detail || error.message))
        this.currentStep = 1
      } finally {
        this.uploading = false
      }
    },

    startStatusCheck() {
      this.checkTaskStatus()
      this.statusCheckInterval = setInterval(() => {
        this.checkTaskStatus()
      }, 2000)
    },

    async checkTaskStatus() {
      if (!this.taskId || this.taskStatus === 'completed') return

      try {
        const response = await apiService.get(
          `${this.apiBaseUrl}/api/av_batch_scan_status/${this.taskId}`
        )

        const data = response.data
        this.taskStatus = data.status
        this.progress = data.progress
        this.scannedFiles = data.scanned_files
        this.elapsedSeconds = data.elapsed_seconds
        this.estimatedRemaining = data.estimated_remaining
        this.currentFileName = data.current_file || ''

        if (this.taskStatus === 'completed') {
          this.stopStatusCheck()
          await this.fetchScanResults()
          this.currentStep = 3
          this.$message.success('检测完成!')
        } else if (this.taskStatus === 'failed') {
          this.stopStatusCheck()
          this.$message.error('检测失败: ' + (data.error || '未知错误'))
        }
      } catch (error) {
        console.error('检查任务状态失败:', error)
      }
    },

    stopStatusCheck() {
      if (this.statusCheckInterval) {
        clearInterval(this.statusCheckInterval)
        this.statusCheckInterval = null
      }
    },

    async fetchScanResults() {
      try {
        const response = await apiService.get(
          `${this.apiBaseUrl}/api/av_batch_scan_result/${this.taskId}`
        )
        this.scanResults = response.data.results
        this.calculateStatistics()
      } catch (error) {
        console.error('获取扫描结果失败:', error)
        this.$message.error('获取扫描结果失败')
      }
    },

    calculateStatistics() {
      this.safeFilesCount = 0
      this.maliciousFilesCount = 0
      this.scanResults.forEach(result => {
        if (result.malicious_count > 0) {
          this.maliciousFilesCount++
        } else {
          this.safeFilesCount++
        }
      })
    },

    async downloadReport() {
      if (!this.taskId) return
      this.downloading = true
      try {
        const response = await apiService.get(
          `${this.apiBaseUrl}/api/av_batch_scan_download/${this.taskId}`,
          { responseType: 'blob' }
        )
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `av_scan_report_${this.taskId}.csv`)
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
        this.$message.success('报告下载成功')
      } catch (error) {
        console.error('下载报告失败:', error)
        this.$message.error('下载报告失败')
      } finally {
        this.downloading = false
      }
    },

    resetBatchScan() {
      this.currentStep = 0
      this.uploadedFiles = []
      this.selectedEngines = []
      this.selectAllEngines = false
      this.taskId = ''
      this.taskStatus = 'pending'
      this.progress = 0
      this.totalFiles = 0
      this.scannedFiles = 0
      this.elapsedSeconds = 0
      this.estimatedRemaining = 0
      this.scanResults = []
      this.currentFileName = ''
      this.stopStatusCheck()
    },

    goToHistory() {
      this.$router.push('/detect/av-scan-history')
    },

    formatFileSize(bytes) {
      if (bytes === 0) return '0 B'
      const k = 1024
      const sizes = ['B', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
    },

    formatTime(seconds) {
      if (!seconds || seconds < 0) return '0秒'
      const h = Math.floor(seconds / 3600)
      const m = Math.floor((seconds % 3600) / 60)
      const s = Math.floor(seconds % 60)
      if (h > 0) return `${h}小时${m}分${s}秒`
      else if (m > 0) return `${m}分${s}秒`
      else return `${s}秒`
    },

    getTaskStatusType(status) {
      const typeMap = { 'pending': 'info', 'running': 'warning', 'completed': 'success', 'failed': 'danger' }
      return typeMap[status] || 'info'
    },

    getTaskStatusText(status) {
      const textMap = { 'pending': '等待中', 'running': '运行中', 'completed': '已完成', 'failed': '失败' }
      return textMap[status] || '未知'
    },

    getEngineStatusType(status) {
      const typeMap = { 'malicious': 'danger', 'safe': 'success', 'unsupported': 'info' }
      return typeMap[status] || 'info'
    },

    getEngineStatusText(status) {
      const textMap = { 'malicious': '恶意', 'safe': '安全', 'unsupported': '不支持' }
      return textMap[status] || 'N/A'
    }
  }
}
</script>

<style scoped>
.av-scan-batch-v2-container {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.steps-container {
  margin: 30px 0;
}

.step-content {
  margin-top: 30px;
}

.upload-section {
  margin: 20px auto;
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
  min-height: 250px;
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

/* 引擎信息区域 */
.engine-info-section {
  margin-top: 30px;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.engine-info-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.engine-info-header h3 {
  margin: 0;
  font-size: 16px;
  color: #303133;
}

.engine-count {
  font-size: 14px;
  color: #909399;
}

.files-list-section {
  margin-top: 30px;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.list-header h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
}

.list-actions {
  display: flex;
  align-items: center;
  gap: 15px;
}

.file-count {
  font-size: 14px;
  color: #606266;
}

.next-step-button {
  text-align: center;
  margin-top: 30px;
}

/* 引擎选择区域 */
.engine-selection-section {
  max-width: 1000px;
  margin: 0 auto;
}

.selection-header {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 20px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
}

.selected-count {
  font-size: 14px;
  color: #606266;
}

.engines-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 15px;
  margin-bottom: 30px;
}

.engine-card {
  border: 2px solid #dcdfe6;
  border-radius: 8px;
  padding: 15px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
  background: white;
}

.engine-card:hover {
  border-color: #409EFF;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
}

.engine-card.selected {
  border-color: #409EFF;
  background: #f0f7ff;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.3);
}

.engine-card.selected .engine-name {
  color: #409EFF;
  font-weight: bold;
}

.engine-icon {
  width: 48px;
  height: 48px;
  margin: 10px auto;
  display: block;
}

.engine-name {
  font-size: 14px;
  color: #303133;
  display: block;
  margin-top: 5px;
}

.action-buttons {
  text-align: center;
  margin-top: 30px;
  display: flex;
  justify-content: center;
  gap: 20px;
}

/* 进度区域 */
.progress-section {
  max-width: 800px;
  margin: 0 auto;
  background: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 25px;
}

.progress-header h3 {
  margin: 0;
  font-size: 20px;
  color: #303133;
}

.progress-bar {
  margin-bottom: 20px;
}

.progress-detail {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-top: 25px;
}

.detail-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.card-icon {
  width: 50px;
  height: 50px;
  background: linear-gradient(135deg, #409EFF 0%, #66b1ff 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.card-icon i {
  font-size: 24px;
  color: white;
}

.card-content {
  flex: 1;
}

.card-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 5px;
}

.card-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}

.card-desc {
  font-size: 12px;
  color: #909399;
}

.file-name-text {
  font-size: 16px;
  word-break: break-all;
}

/* 结果区域 */
.result-section {
  background: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 25px;
}

.result-header h3 {
  margin: 0;
  font-size: 20px;
  color: #303133;
}

.result-actions {
  display: flex;
  align-items: center;
  gap: 15px;
}

.result-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.summary-item {
  background: white;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.summary-icon {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.summary-icon i {
  font-size: 28px;
  color: white;
}

.summary-content {
  flex: 1;
}

.summary-value {
  font-size: 32px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 5px;
}

.summary-label {
  font-size: 14px;
  color: #909399;
}

.result-table-container {
  overflow-x: auto;
}

.malicious-count {
  color: #f56c6c;
  font-weight: bold;
}
</style>
