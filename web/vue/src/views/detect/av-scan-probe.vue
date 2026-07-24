<template>
  <div class="av-scan-probe-container">
    <!-- 标题 -->
    <div class="text-center">
      <h2 class="text-primary">边界探测</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG — 二分查找定位杀毒引擎检测边界</p>
    </div>

    <!-- 步骤指示器 -->
    <el-steps :active="currentStep" finish-status="success" simple class="steps-container">
      <el-step title="上传文件" icon="el-icon-upload" />
      <el-step title="选择引擎" icon="el-icon-setting" />
      <el-step title="探测进度" icon="el-icon-video-play" />
      <el-step title="探测结果" icon="el-icon-document" />
    </el-steps>

    <!-- ========= 第一步：上传文件 ========= -->
    <div v-show="currentStep === 0" class="step-content">
      <div class="upload-section">
        <input
          ref="fileInput"
          class="file-upload-input"
          type="file"
          @change="handleFileChange"
        />
        <div class="drop-zone" @drop="handleDrop" @dragover.prevent>
          <div v-if="!uploading">
            <svg-icon icon-class="upload" class="upload-icon" />
            <p class="drop-text">把待检测文件拖到这里或</p>
            <el-button type="primary" size="large" @click="handleUpload">选择待检测文件</el-button>
            <p class="drop-hint">仅支持单个文件</p>
          </div>
          <div v-else class="uploading-state">
            <i class="el-icon-loading" style="font-size: 48px; color: #409EFF;" />
            <p class="uploading-text">正在准备文件...</p>
          </div>
        </div>
      </div>

      <!-- 引擎列表 -->
      <div class="engine-info-section">
        <div class="engine-info-header">
          <h3>支持检测引擎列表</h3>
          <span class="engine-count">共 {{ engineList.length }} 个引擎</span>
        </div>
        <el-table :data="engineList" border style="width: 100%" size="small">
          <el-table-column type="index" label="序号" width="60" align="center" />
          <el-table-column prop="name" label="引擎名称" align="center" />
          <el-table-column prop="vm" label="所在虚拟机" align="center" />
        </el-table>
      </div>

      <!-- 已选文件 -->
      <div v-if="selectedFile" class="file-card">
        <div class="file-info">
          <i class="el-icon-document" style="font-size:32px;color:#409EFF;" />
          <div class="file-detail">
            <span class="file-name">{{ selectedFile.name }}</span>
            <span class="file-size">{{ selectedFile.sizeStr }}</span>
          </div>
          <el-button type="danger" size="small" @click="clearFile">移除</el-button>
        </div>
        <div class="next-step-button">
          <el-button type="primary" size="large" @click="currentStep = 1">下一步：选择引擎</el-button>
        </div>
      </div>
    </div>

    <!-- ========= 第二步：选择引擎 ========= -->
    <div v-show="currentStep === 1" class="step-content">
      <div class="engine-selection-section">
        <div class="selection-header">
          <el-checkbox v-model="selectAllEngines" @change="handleSelectAll">全选</el-checkbox>
          <span class="selected-count">已选择 {{ selectedEngines.length }} / {{ allEngines.length }} 个引擎</span>
        </div>
        <div class="engines-grid">
          <div
            v-for="engine in allEngines"
            :key="engine"
            class="engine-card"
            :class="{ selected: selectedEngines.includes(engine) }"
            @click="toggleEngine(engine)"
          >
            <img :src="getEngineIcon(engine)" class="engine-icon" @error="handleIconError" />
            <span class="engine-name">{{ engine }}</span>
          </div>
        </div>
        <div class="action-buttons">
          <el-button @click="currentStep = 0">上一步</el-button>
          <el-button type="primary" :disabled="selectedEngines.length === 0" @click="startProbe">
            开始边界探测
          </el-button>
        </div>
      </div>
    </div>

    <!-- ========= 第三步：探测进度 ========= -->
    <div v-show="currentStep === 2" class="step-content">
      <div class="progress-section">
        <!-- 排队提示 -->
        <el-alert
          v-if="queuePosition > 1"
          :title="queueMessage || `排队中，前面还有 ${queuePosition - 1} 个任务`"
          type="warning"
          :closable="false"
          show-icon
          style="margin-bottom: 20px;"
        />
        <!-- 总体进度 -->

        <div class="overall-progress" v-if="queuePosition <= 1">
          <el-progress :percentage="overallProgress" :stroke-width="20" color="#409EFF" />
          <p class="progress-hint">
            已完成 {{ completedEngines }} / {{ selectedEngines.length }} 个引擎
          </p>
        </div>

        <!-- 每个引擎的实时状态 -->
        <div class="engines-progress">
          <div
            v-for="ep in engineProgress"
            :key="ep.engine"
            class="engine-progress-card"
            :class="engineCardClass(ep)"
          >
            <div class="ep-header">
              <span class="ep-engine-name">{{ ep.engine }}</span>
              <el-tag :type="engineTagType(ep)" size="small">{{ ep.stateText }}</el-tag>
            </div>
            <div v-if="ep.state === 'bisecting'" class="ep-body">
              <div class="ep-range">
                搜索范围: <code>0x{{ ep.lowHex }} ~ 0x{{ ep.highHex }}</code>
                ({{ ep.low }} ~ {{ ep.high }})
              </div>
              <div class="ep-mid">
                当前切分点: <code>0x{{ ep.midHex }}</code> → {{ ep.lastDetected ? '检出' : '未检出' }}
              </div>
            </div>
            <div class="ep-stats">
              <span>查询次数: {{ ep.total_queries }}</span>
              <span>已找到签名: {{ ep.found_signatures }}</span>
              <span v-if="ep.intervals.length > 0">
                区间: {{ ep.intervals.map(i => `[${i[0]},${i[1]}]`).join(', ') }}
              </span>
            </div>
            <div v-if="ep.intervals.length > 0" class="ep-intervals">
              <el-tag
                v-for="(iv, idx) in ep.intervals"
                :key="idx"
                size="small"
                type="warning"
                style="margin: 2px"
              >
                {{ iv[0] }}–{{ iv[1] }}
              </el-tag>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ========= 第四步：探测结果 ========= -->
    <div v-show="currentStep === 3" class="step-content">
      <div class="results-section">
        <h3>探测结果汇总</h3>
        <el-table :data="probeResults" border style="width: 100%">
          <el-table-column prop="engine" label="引擎" width="140" align="center" />
          <el-table-column label="找到签名数" width="100" align="center">
            <template #default="{ row }">{{ row.found_signatures }}</template>
          </el-table-column>
          <el-table-column label="总查询次数" width="110" align="center">
            <template #default="{ row }">{{ row.total_queries }}</template>
          </el-table-column>
          <el-table-column label="边界区间 (hex)" min-width="300">
            <template #default="{ row }">
              <template v-if="row.intervals && row.intervals.length > 0">
                <el-tag
                  v-for="(iv, idx) in row.intervals"
                  :key="idx"
                  size="small"
                  style="margin:2px;font-family:monospace;"
                >
                  {{ iv[0] }} – {{ iv[1] }}
                </el-tag>
              </template>
              <span v-else style="color:#999">未触发检测</span>
            </template>
          </el-table-column>
          <el-table-column label="状态" width="90" align="center">
            <template #default="{ row }">
              <el-tag v-if="row.state === 'done'" type="success" size="small">完成</el-tag>
              <el-tag v-else-if="row.error" type="danger" size="small">错误</el-tag>
              <el-tag v-else size="small">{{ row.state }}</el-tag>
            </template>
          </el-table-column>
        </el-table>

        <div class="action-buttons" style="margin-top:20px;">
          <el-button @click="resetAll">重新探测</el-button>
          <el-button type="primary" @click="exportCSV">导出 CSV</el-button>
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
  name: 'AVScanProbe',
  data() {
    return {
      currentStep: 0,
      uploading: false,
      selectedFile: null,
      probeRunning: false,
      engineList: [],
      allEngines: [],
      selectedEngines: [],
      selectAllEngines: false,
      apiBaseUrl: import.meta.env.VITE_APP_BASE_API || '',

      // 进度
      completedEngines: 0,
      engineProgress: [],
      probeResults: [],
      // 排队
      queuePosition: 0,
      queueMessage: '',
    }
  },
  computed: {
    overallProgress() {
      if (this.selectedEngines.length === 0) return 0
      return Math.round((this.completedEngines / this.selectedEngines.length) * 100)
    },
  },
  async mounted() {
    await this.fetchEngines()
    this.checkSavedProbe()
  },
  methods: {
    // ======== 引擎列表 ========
    async fetchEngines() {
      try {
        const resp = await apiService.get(`${this.apiBaseUrl}/api/av_engines`)
        const data = resp.data
        this.engineList = data.engines || []
        this.allEngines = this.engineList.map(e => e.name)
      } catch (e) {
        console.error('获取引擎列表失败:', e)
        this.$message.error('获取引擎列表失败')
      }
    },

    // ======== 文件操作 ========
    handleUpload() {
      this.$refs.fileInput.click()
    },
    handleFileChange(e) {
      const file = e.target.files[0]
      if (file) this.setFile(file)
    },
    handleDrop(e) {
      const file = e.dataTransfer.files[0]
      if (file) this.setFile(file)
    },
    setFile(file) {
      const sizeMB = file.size / (1024 * 1024)
      this.selectedFile = {
        name: file.name,
        size: file.size,
        sizeStr: sizeMB >= 1 ? `${sizeMB.toFixed(2)} MB` : `${file.size} 字节`,
        raw: file,
      }
    },
    clearFile() {
      this.selectedFile = null
      if (this.$refs.fileInput) this.$refs.fileInput.value = ''
    },

    // ======== 引擎选择 ========
    toggleEngine(engine) {
      const idx = this.selectedEngines.indexOf(engine)
      if (idx >= 0) {
        this.selectedEngines.splice(idx, 1)
      } else {
        this.selectedEngines.push(engine)
      }
      this.selectAllEngines = this.selectedEngines.length === this.allEngines.length
    },
    handleSelectAll(val) {
      this.selectedEngines = val ? [...this.allEngines] : []
    },
    getEngineIcon(name) {
      try {
        return require(`@/assets/engine/${name}.png`)
      } catch {
        return ''
      }
    },
    handleIconError(e) {
      e.target.style.display = 'none'
    },

    // ======== 开始探测 ========
    async startProbe() {
      this.currentStep = 2
      this.completedEngines = 0
      this.probeResults = []
      // 保存任务信息，刷新后可恢复
      localStorage.setItem('av_probe_file', JSON.stringify({
        name: this.selectedFile.name, size: this.selectedFile.size,
        engines: this.selectedEngines,
        startedAt: Date.now(),
      }))
      // 初始化每个引擎的进度对象
      this.engineProgress = this.selectedEngines.map(engine => ({
        engine,
        state: 'waiting',
        stateText: '等待中',
        low: 0,
        high: 0,
        mid: 0,
        lowHex: '0x0',
        highHex: '0x0',
        midHex: '0x0',
        iteration: 0,
        total_queries: 0,
        found_signatures: 0,
        intervals: [],
        lastDetected: null,
        error: '',
      }))

      try {
        const formData = new FormData()
        formData.append('file', this.selectedFile.raw)
        formData.append('engines', this.selectedEngines.join(','))

        const token = localStorage.getItem('token') || sessionStorage.getItem('token')
        const response = await fetch(`${this.apiBaseUrl}/api/av_probe_start`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
          body: formData,
        })

        if (!response.ok) {
          const err = await response.json()
          throw new Error(err.detail || `HTTP ${response.status}`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''   // 保留未完成的行

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event = JSON.parse(line.slice(6))
                this.handleSSEEvent(event)
              } catch {
                // 忽略解析错误
              }
            }
          }
        }
      } catch (err) {
        this.$message.error(`探测失败: ${err.message}`)
      }
    },
    handleSSEEvent(event) {
      const { type, engine } = event

      // 找到对应引擎的进度对象
      const ep = this.engineProgress.find(p => p.engine === engine)

      switch (type) {
        case 'queued':
          this.queuePosition = event.position || 0
          this.queueMessage = event.message || ''
          break

        case 'start':
          this.queuePosition = 0
          this.queueMessage = ''
          break

        case 'engine_start':
          if (ep) {
            ep.state = 'running'
            ep.stateText = '初始化'
          }
          break

        case 'iteration':
          if (ep) {
            Object.assign(ep, {
              state: event.state,
              stateText: this.stateLabel(event.state),
              low: event.low,
              high: event.high,
              mid: event.mid,
              lowHex: '0x' + event.low.toString(16).toUpperCase(),
              highHex: '0x' + event.high.toString(16).toUpperCase(),
              midHex: '0x' + event.mid.toString(16).toUpperCase(),
              iteration: event.iteration,
              total_queries: event.total_queries,
              found_signatures: event.found_signatures,
              intervals: event.intervals_hex || [],
              lastDetected: event.detected,
            })
          }
          break

        case 'boundary_found':
          if (ep) {
            ep.intervals = event.intervals || []
            ep.found_signatures = event.signature_count
          }
          break

        case 'engine_done':
          if (ep) {
            ep.state = 'done'
            ep.stateText = '完成'
            ep.found_signatures = event.found_signatures
            ep.total_queries = event.total_queries
            ep.intervals = event.intervals || []
            ep.error = event.error || ''
          }
          this.completedEngines++
          // 存结果
          this.probeResults.push({
            engine: engine,
            state: 'done',
            found_signatures: event.found_signatures,
            total_queries: event.total_queries,
            intervals: event.intervals || [],
            error: event.error || '',
          })
          break

        case 'engine_error':
          if (ep) {
            ep.state = 'error'
            ep.stateText = '错误'
            ep.error = event.error
          }
          this.completedEngines++
          this.probeResults.push({
            engine: engine,
            state: 'error',
            found_signatures: 0,
            total_queries: 0,
            intervals: [],
            error: event.error,
          })
          break

        case 'complete':
          this.currentStep = 3
          localStorage.removeItem('av_probe_file')
          this.$message.success('边界探测完成')
          break

        case 'error':
          localStorage.removeItem('av_probe_file')
          this.$message.error(`探测异常: ${event.error}`)
          break
      }
    },

    // ======== 辅助方法 ========
    stateLabel(state) {
      const map = {
        initial: '初始化',
        bisecting: '二分查找',
        masking_check: '掩码验证',
        masking: '掩码处理',
        clean: '未检出',
        done: '完成',
        waiting: '等待中',
        running: '运行中',
      }
      return map[state] || state
    },
    engineCardClass(ep) {
      if (ep.state === 'done') return 'engine-done'
      if (ep.state === 'error') return 'engine-error'
      if (ep.state === 'bisecting' || ep.state === 'masking_check') return 'engine-active'
      return ''
    },
    engineTagType(ep) {
      if (ep.state === 'done') return 'success'
      if (ep.state === 'error') return 'danger'
      if (ep.state === 'bisecting' || ep.state === 'masking_check') return ''
      return 'info'
    },

    // ======== 结果操作 ========
    exportCSV() {
      let csv = '引擎,签名数量,总查询次数,边界区间(hex),错误\n'
      for (const r of this.probeResults) {
        const intervals = (r.intervals || []).map(i => `${i[0]}-${i[1]}`).join('; ')
        csv += `${r.engine},${r.found_signatures},${r.total_queries},"${intervals}","${r.error || ''}"\n`
      }
      const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `probe_result_${Date.now()}.csv`
      a.click()
      URL.revokeObjectURL(url)
    },
    resetAll() {
      this.currentStep = 0
      this.selectedFile = null
      this.selectedEngines = []
      this.engineProgress = []
      this.probeResults = []
      this.completedEngines = 0
      localStorage.removeItem('av_probe_file')
    },

    checkSavedProbe() {
      const saved = localStorage.getItem('av_probe_file')
      if (!saved) return
      try {
        const info = JSON.parse(saved)
        const elapsed = Math.floor((Date.now() - info.startedAt) / 1000)
        this.$notify({
          title: '检测到未完成任务',
          message: `${info.name} 正在边界探测中 (已运行 ${elapsed}秒)，刷新页面不会中断任务。请等待完成后重新探测。`,
          type: 'warning',
          duration: 8000,
        })
      } catch {}
    },
  },
}
</script>

<style scoped>
.av-scan-probe-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}
.text-center { text-align: center; margin-bottom: 20px; }
.text-primary { color: #303133; }
.text-secondary { color: #909399; font-size: 14px; }
.steps-container { margin-bottom: 30px; }

/* ====== 上传区 ====== */
.drop-zone {
  border: 2px dashed #dcdfe6;
  border-radius: 8px;
  padding: 50px 20px;
  text-align: center;
  background: #fafafa;
  transition: border-color 0.3s;
}
.drop-zone:hover { border-color: #409EFF; }
.upload-icon { font-size: 48px; color: #c0c4cc; }
.drop-text { color: #606266; margin: 12px 0; }
.drop-hint { color: #c0c4cc; font-size: 12px; margin-top: 8px; }
.file-upload-input { display: none; }

/* ====== 引擎信息 ====== */
.engine-info-section { margin-top: 30px; }
.engine-info-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.engine-count { color: #909399; font-size: 13px; }

/* ====== 文件卡片 ====== */
.file-card {
  margin-top: 20px;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 16px;
}
.file-info { display: flex; align-items: center; gap: 12px; }
.file-detail { flex: 1; }
.file-name { display: block; font-weight: 600; }
.file-size { display: block; color: #909399; font-size: 12px; margin-top: 4px; }
.next-step-button { text-align: right; margin-top: 16px; }

/* ====== 引擎选择 ====== */
.engine-selection-section { padding: 10px 0; }
.selection-header { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.selected-count { color: #606266; }
.engines-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.engine-card {
  width: 120px;
  padding: 12px 8px;
  border: 2px solid #e4e7ed;
  border-radius: 8px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}
.engine-card:hover { border-color: #409EFF; }
.engine-card.selected { border-color: #409EFF; background: #ecf5ff; }
.engine-icon { width: 40px; height: 40px; margin-bottom: 6px; }
.engine-name { display: block; font-size: 12px; color: #303133; }
.action-buttons { display: flex; gap: 12px; justify-content: center; margin-top: 20px; }

/* ====== 进度区 ====== */
.overall-progress { margin-bottom: 24px; }
.progress-hint { text-align: center; color: #909399; margin-top: 8px; }
.engines-progress { display: flex; flex-direction: column; gap: 10px; }
.engine-progress-card {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 12px 16px;
  background: #fff;
}
.engine-progress-card.engine-active { border-color: #409EFF; background: #ecf5ff; }
.engine-progress-card.engine-done { border-color: #67c23a; background: #f0f9eb; }
.engine-progress-card.engine-error { border-color: #f56c6c; background: #fef0f0; }
.ep-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.ep-engine-name { font-weight: 600; }
.ep-body { font-size: 13px; color: #606266; }
.ep-body code { background: #f0f2f5; padding: 1px 6px; border-radius: 3px; font-family: Consolas, monospace; }
.ep-mid { margin-top: 4px; }
.ep-stats { display: flex; gap: 20px; font-size: 12px; color: #909399; margin-top: 6px; }

/* ====== 结果区 ====== */
.results-section h3 { margin-bottom: 16px; }
</style>
