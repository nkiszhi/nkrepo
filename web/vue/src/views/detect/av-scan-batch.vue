<template>
  <div class="av-scan-batch-container">
    <!-- 标题 -->
    <div class="text-center">
      <h2 class="text-primary">批量杀毒软件检测</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
    </div>

    <!-- 步骤指示器 -->
    <el-steps :active="currentStep" finish-status="success" simple class="steps-container">
      <el-step title="上传样本" icon="el-icon-upload"></el-step>
      <el-step title="开始检测" icon="el-icon-video-play"></el-step>
      <el-step title="查看结果" icon="el-icon-document"></el-step>
    </el-steps>

    <!-- 第一步:上传样本 -->
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
            <el-progress 
              :percentage="uploadProgress" 
              :stroke-width="10"
              class="upload-progress"
            ></el-progress>
          </div>
        </div>
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
              <el-button
                type="text"
                size="small"
                @click="removeFile(scope.$index)"
              >
                删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>

        <div class="next-step-button">
          <el-button 
            type="primary" 
            size="large" 
            @click="startBatchScan"
            :loading="uploading"
          >
            {{ uploading ? '上传中...' : '开始批量检测' }}
          </el-button>
        </div>
      </div>
    </div>

    <!-- 第二步:检测进度 -->
    <div v-show="currentStep === 1" class="step-content">
      <div class="progress-section">
        <div class="progress-header">
          <h3>{{ progressTitle }}</h3>
          <el-tag :type="getTaskStatusType(taskStatus)" size="large">
            {{ getTaskStatusText(taskStatus) }}
          </el-tag>
        </div>

        <!-- 当前步骤说明 -->
        <div class="current-step-info">
          <div class="step-indicator">
            <i :class="currentStepIcon" class="step-icon"></i>
            <span class="step-text">{{ currentStepText }}</span>
          </div>
        </div>

        <!-- 进度条 -->
        <el-progress
          :percentage="progress"
          :status="progressStatus"
          :stroke-width="25"
          :show-text="true"
          class="progress-bar"
        >
          <template #default="{ percentage }">
            <span class="progress-text">{{ percentage }}%</span>
          </template>
        </el-progress>

        <!-- 详细进度信息 -->
        <div class="progress-detail">
          <div class="detail-card">
            <div class="card-icon">
              <i class="el-icon-document"></i>
            </div>
            <div class="card-content">
              <div class="card-label">文件进度</div>
              <div class="card-value">{{ scannedFiles }} / {{ totalFiles }}</div>
              <div class="card-desc">已完成扫描</div>
            </div>
          </div>

          <div class="detail-card">
            <div class="card-icon">
              <i class="el-icon-time"></i>
            </div>
            <div class="card-content">
              <div class="card-label">已用时间</div>
              <div class="card-value">{{ formatTime(elapsedSeconds) }}</div>
              <div class="card-desc">预计剩余: {{ formatTime(estimatedRemaining) }}</div>
            </div>
          </div>

          <div class="detail-card" v-if="currentFileName">
            <div class="card-icon">
              <i class="el-icon-loading"></i>
            </div>
            <div class="card-content">
              <div class="card-label">当前文件</div>
              <div class="card-value file-name-text">{{ currentFileName }}</div>
              <div class="card-desc">正在检测中...</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 第三步:检测结果 -->
    <div v-show="currentStep === 2" class="step-content">
      <div class="result-section">
        <div class="result-header">
          <h3>检测结果</h3>
          <div class="result-actions">
            <el-tag type="success" size="large">
              <i class="el-icon-success"></i>
              检测完成
            </el-tag>
            <el-button type="success" @click="downloadReport" :loading="downloading">
              <svg-icon icon-class="download" style="margin-right: 8px;" />
              下载CSV报告
            </el-button>
          </div>
        </div>

        <!-- 统计信息 -->
        <div class="result-summary">
          <div class="summary-item">
            <div class="summary-icon" style="background: #67c23a;">
              <i class="el-icon-check"></i>
            </div>
            <div class="summary-content">
              <div class="summary-value">{{ safeFilesCount }}</div>
              <div class="summary-label">安全文件</div>
            </div>
          </div>
          <div class="summary-item">
            <div class="summary-icon" style="background: #f56c6c;">
              <i class="el-icon-warning"></i>
            </div>
            <div class="summary-content">
              <div class="summary-value">{{ maliciousFilesCount }}</div>
              <div class="summary-label">恶意文件</div>
            </div>
          </div>
          <div class="summary-item">
            <div class="summary-icon" style="background: #409EFF;">
              <i class="el-icon-document"></i>
            </div>
            <div class="summary-content">
              <div class="summary-value">{{ totalFiles }}</div>
              <div class="summary-label">总文件数</div>
            </div>
          </div>
        </div>

        <!-- 结果表格 -->
        <div class="result-table-container">
          <el-table
            :data="scanResults"
            border
            style="width: 100%"
            max-height="600"
          >
            <el-table-column prop="file_name" label="文件名" fixed width="200"></el-table-column>
            <el-table-column
              v-for="engine in avEngines"
              :key="engine"
              :prop="`engines.${engine}`"
              :label="engine"
              width="100"
            >
              <template #default="scope">
                <el-tag
                  :type="getEngineStatusType(scope.row.engines[engine])"
                  size="small"
                >
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
          <el-button type="primary" @click="resetBatchScan">
            继续检测
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

// 创建axios实例
const apiService = axios.create({
  timeout: 600000,
  headers: {
    'Content-Type': 'application/json'
  }
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
  name: 'AVScanBatch',
  data() {
    return {
      currentStep: 0,
      uploadedFiles: [],
      uploading: false,
      uploadProgress: 0,
      taskId: '',
      taskStatus: 'pending',
      progress: 0,
      totalFiles: 0,
      scannedFiles: 0,
      elapsedSeconds: 0,
      estimatedRemaining: 0,
      checkingStatus: false,
      downloading: false,
      scanResults: [],
      apiBaseUrl: '', // 初始为空,等待从配置文件加载
      statusCheckInterval: null,
      currentFileName: '',
      safeFilesCount: 0,
      maliciousFilesCount: 0,
      errorCount: 0, // 错误计数器
      maxErrorCount: 3, // 最大错误次数
      isFetchingResults: false, // 是否正在获取结果
      avEngines: [
        'Avira', 'McAfee', 'WindowsDefender', 'IkarusT3', 'Emsisoft',
        'FProtect', 'Vba32', 'ClamAV', 'Kaspersky', 'ESET',
        'DrWeb', 'Avast', 'AVG', 'AdAware', 'FSecure'
      ]
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
    },
    currentStepIcon() {
      if (this.taskStatus === 'pending') return 'el-icon-loading'
      if (this.taskStatus === 'running') return 'el-icon-video-play'
      if (this.taskStatus === 'completed') return 'el-icon-success'
      if (this.taskStatus === 'failed') return 'el-icon-error'
      return 'el-icon-loading'
    },
    currentStepText() {
      if (this.taskStatus === 'pending') return '正在初始化检测任务...'
      if (this.taskStatus === 'running') {
        if (this.scannedFiles === 0) {
          return '开始检测样本文件...'
        }
        return `正在检测第 ${this.scannedFiles + 1} 个文件,共 ${this.totalFiles} 个`
      }
      if (this.taskStatus === 'completed') return '所有文件检测完成!'
      if (this.taskStatus === 'failed') return '检测过程中出现错误'
      return '准备检测...'
    }
  },
  created() {
    this.loadConfig()
    // 检查是否有未完成的任务
    this.checkUnfinishedTask()
  },
  beforeDestroy() {
    // 清除定时器
    if (this.statusCheckInterval) {
      clearInterval(this.statusCheckInterval)
    }
  },
  methods: {
    // 加载配置
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
              break
            }
          }
          if (inApiSection && trimmedLine.startsWith('[')) {
            break
          }
        }
      } catch (error) {
        console.warn('加载配置文件失败:', error.message)
      }
    },

    // 检查未完成任务
    async checkUnfinishedTask() {
      const savedTaskId = localStorage.getItem('av_batch_task_id')
      if (savedTaskId) {
        // 先验证任务是否存在
        try {
          const response = await apiService.get(
            `${this.apiBaseUrl}/api/av_batch_scan_status/${savedTaskId}`
          )

          // 任务存在,恢复状态
          this.taskId = savedTaskId
          this.currentStep = 1
          this.taskStatus = response.data.status
          this.progress = response.data.progress
          this.totalFiles = response.data.total_files
          this.scannedFiles = response.data.scanned_files
          this.currentFileName = response.data.current_file
          this.elapsedSeconds = response.data.elapsed_seconds
          this.estimatedRemaining = response.data.estimated_remaining

          // 如果任务还在运行,开始轮询
          if (this.taskStatus === 'running' || this.taskStatus === 'pending') {
            this.startStatusCheck()
          } else if (this.taskStatus === 'completed') {
            // 如果已完成,获取结果
            await this.fetchScanResults()
            this.currentStep = 2
          }
        } catch (error) {
          // 任务不存在(后端重启了),清除localStorage
          console.warn('保存的任务不存在:', error.message)
          localStorage.removeItem('av_batch_task_id')
        }
      }
    },

    // 文件拖拽处理
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
      // 清空input
      e.target.value = ''
    },

    // 添加文件
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

    // 删除文件
    removeFile(index) {
      this.uploadedFiles.splice(index, 1)
    },

    // 清空所有文件
    clearAllFiles() {
      this.uploadedFiles = []
    },

    // 开始批量扫描
    async startBatchScan() {
      if (this.uploadedFiles.length === 0) {
        this.$message.error('请先上传文件')
        return
      }

      this.uploading = true
      this.uploadProgress = 0

      try {
        // 上传文件
        const formData = new FormData()
        this.uploadedFiles.forEach(file => {
          formData.append('files', file.file)
        })

        const uploadResponse = await apiService.post(
          `${this.apiBaseUrl}/api/av_batch_upload`,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            onUploadProgress: (progressEvent) => {
              // 计算上传进度
              if (progressEvent.total) {
                this.uploadProgress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
              }
            }
          }
        )

        this.taskId = uploadResponse.data.task_id
        this.totalFiles = uploadResponse.data.total_files

        // 保存任务ID到localStorage
        localStorage.setItem('av_batch_task_id', this.taskId)

        this.$message.success(`成功上传 ${this.totalFiles} 个文件,正在启动检测...`)

        // 立即跳转到步骤1(检测进度)
        this.currentStep = 1
        this.taskStatus = 'pending'

        // 立即开始轮询状态(不需要等待后端响应)
        this.startStatusCheck()

        // 启动批量扫描(异步,不等待)
        apiService.post(`${this.apiBaseUrl}/api/av_batch_scan_start`, {
          task_id: this.taskId
        }).then(response => {
          console.log('启动扫描响应:', response.data)
          this.$message.success('批量检测任务已启动')
        }).catch(error => {
          console.error('启动扫描失败:', error)
          this.stopStatusCheck()
          this.$message.error('启动扫描失败: ' + (error.response?.data?.detail || error.message))
          this.currentStep = 0
        })

      } catch (error) {
        console.error('启动批量扫描失败:', error)
        let errMsg = '启动批量扫描失败'
        if (error.response?.data?.detail) {
          errMsg = error.response.data.detail
        }
        this.$message.error(errMsg)
        this.currentStep = 0
      } finally {
        this.uploading = false
      }
    },

    // 开始状态检查
    startStatusCheck() {
      console.log('=== 开始轮询状态 ===')
      console.log('taskId:', this.taskId)

      // 立即检查一次
      this.checkTaskStatus()

      // 每2秒检查一次
      this.statusCheckInterval = setInterval(() => {
        this.checkTaskStatus()
      }, 2000)
    },

    // 检查任务状态
    async checkTaskStatus() {
      if (!this.taskId) return

      // 如果已经完成,不再查询
      if (this.taskStatus === 'completed' || this.isFetchingResults) {
        return
      }

      try {
        const response = await apiService.get(
          `${this.apiBaseUrl}/api/av_batch_scan_status/${this.taskId}`
        )

        const data = response.data
        console.log('任务状态响应:', data)

        this.taskStatus = data.status
        this.progress = data.progress
        this.scannedFiles = data.scanned_files
        this.elapsedSeconds = data.elapsed_seconds
        this.estimatedRemaining = data.estimated_remaining

        // 成功查询,重置错误计数
        this.errorCount = 0

        // 更新当前文件名
        if (data.current_file) {
          this.currentFileName = data.current_file
        }

        // 如果任务完成,停止检查并获取结果
        if (this.taskStatus === 'completed') {
          this.stopStatusCheck()

          // 防止重复获取结果
          if (!this.isFetchingResults) {
            this.isFetchingResults = true
            await this.fetchScanResults()
            this.currentStep = 2
            // 清除localStorage
            localStorage.removeItem('av_batch_task_id')
            this.$message.success('检测完成!')
          }
          return // 立即返回,不再继续执行
        } else if (this.taskStatus === 'failed') {
          this.stopStatusCheck()
          this.$message.error('批量检测任务失败: ' + (data.error || '未知错误'))
          return // 立即返回
        }

      } catch (error) {
        console.error('检查任务状态失败:', error)
        this.errorCount++

        // 如果是404错误,说明任务不存在(后端重启了)
        if (error.response && error.response.status === 404) {
          this.stopStatusCheck()
          localStorage.removeItem('av_batch_task_id')
          this.$message.warning('任务不存在,可能后端已重启,请重新上传文件')
          this.currentStep = 0
          this.taskId = ''
          this.taskStatus = 'pending'
        } else if (this.errorCount >= this.maxErrorCount) {
          // 连续错误超过3次,停止查询
          this.stopStatusCheck()
          this.$message.error('连续查询失败,已停止自动更新')
        }
      }
    },

    // 停止状态检查
    stopStatusCheck() {
      if (this.statusCheckInterval) {
        clearInterval(this.statusCheckInterval)
        this.statusCheckInterval = null
      }
    },

    // 获取扫描结果
    async fetchScanResults() {
      try {
        const response = await apiService.get(
          `${this.apiBaseUrl}/api/av_batch_scan_result/${this.taskId}`
        )

        this.scanResults = response.data.results

        // 计算统计信息
        this.calculateStatistics()

      } catch (error) {
        console.error('获取扫描结果失败:', error)
        this.$message.error('获取扫描结果失败')
      }
    },

    // 计算统计信息
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

    // 下载报告
    async downloadReport() {
      if (!this.taskId) return

      this.downloading = true

      try {
        const response = await apiService.get(
          `${this.apiBaseUrl}/api/av_batch_scan_download/${this.taskId}`,
          {
            responseType: 'blob'
          }
        )

        // 创建下载链接
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

    // 重置批量扫描
    resetBatchScan() {
      this.currentStep = 0
      this.uploadedFiles = []
      this.taskId = ''
      this.taskStatus = 'pending'
      this.progress = 0
      this.totalFiles = 0
      this.scannedFiles = 0
      this.elapsedSeconds = 0
      this.estimatedRemaining = 0
      this.scanResults = []
      this.currentFileName = ''
      this.isFetchingResults = false
      this.errorCount = 0
      this.stopStatusCheck()
      localStorage.removeItem('av_batch_task_id')
    },

    // 格式化文件大小
    formatFileSize(bytes) {
      if (bytes === 0) return '0 B'
      const k = 1024
      const sizes = ['B', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
    },

    // 格式化时间
    formatTime(seconds) {
      if (!seconds || seconds < 0) return '0秒'
      const h = Math.floor(seconds / 3600)
      const m = Math.floor((seconds % 3600) / 60)
      const s = Math.floor(seconds % 60)

      if (h > 0) {
        return `${h}小时${m}分${s}秒`
      } else if (m > 0) {
        return `${m}分${s}秒`
      } else {
        return `${s}秒`
      }
    },

    // 获取任务状态类型
    getTaskStatusType(status) {
      const typeMap = {
        'pending': 'info',
        'running': 'warning',
        'completed': 'success',
        'failed': 'danger'
      }
      return typeMap[status] || 'info'
    },

    // 获取任务状态文本
    getTaskStatusText(status) {
      const textMap = {
        'pending': '等待中',
        'running': '运行中',
        'completed': '已完成',
        'failed': '失败'
      }
      return textMap[status] || '未知'
    },

    // 获取引擎状态类型
    getEngineStatusType(status) {
      const typeMap = {
        'malicious': 'danger',
        'safe': 'success',
        'unsupported': 'info'
      }
      return typeMap[status] || 'info'
    },

    // 获取引擎状态文本
    getEngineStatusText(status) {
      const textMap = {
        'malicious': '恶意',
        'safe': '安全',
        'unsupported': '不支持'
      }
      return textMap[status] || 'N/A'
    }
  }
}
</script>

<style scoped>
.av-scan-batch-container {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

/* 步骤指示器 */
.steps-container {
  margin: 30px 0;
}

/* 步骤内容 */
.step-content {
  margin-top: 30px;
}

/* 上传区域 */
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

.uploading-state {
  text-align: center;
  width: 100%;
}

.uploading-text {
  margin-top: 20px;
  font-size: 16px;
  color: #606266;
}

.upload-progress {
  margin-top: 20px;
  max-width: 400px;
  margin-left: auto;
  margin-right: auto;
}

/* 文件列表 */
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

.current-step-info {
  margin-bottom: 25px;
  padding: 20px;
  background: linear-gradient(135deg, #f0f7ff 0%, #e6f1ff 100%);
  border-radius: 8px;
  border-left: 4px solid #409EFF;
}

.step-indicator {
  display: flex;
  align-items: center;
  gap: 15px;
}

.step-icon {
  font-size: 32px;
  color: #409EFF;
}

.step-text {
  font-size: 18px;
  font-weight: bold;
  color: #303133;
}

.progress-info {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-bottom: 25px;
}

.info-item {
  display: flex;
  flex-direction: column;
}

.info-item .label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 5px;
}

.info-item .value {
  font-size: 18px;
  font-weight: bold;
  color: #303133;
}

.progress-bar {
  margin-bottom: 20px;
}

.progress-text {
  font-size: 16px;
  font-weight: bold;
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
  transition: all 0.3s;
}

.detail-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
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
  margin-bottom: 5px;
}

.card-desc {
  font-size: 12px;
  color: #909399;
}

.file-name-text {
  font-size: 16px;
  word-break: break-all;
}

.detail-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #606266;
}

.detail-item i {
  font-size: 16px;
  color: #409EFF;
}

.time-info {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  margin-bottom: 25px;
}

.time-item {
  display: flex;
  flex-direction: column;
}

.time-item .label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 5px;
}

.time-item .value {
  font-size: 16px;
  color: #606266;
}

.progress-actions {
  text-align: center;
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
  transition: all 0.3s;
}

.summary-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
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

.result-actions {
  text-align: center;
  margin-top: 30px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .progress-info {
    grid-template-columns: 1fr;
  }

  .time-info {
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
