<template>
  <div class="dolos-analysis-container">
    <el-card class="main-card">
      <template #header>
        <div class="card-header">
          <span class="title">代码相似度
          检测分析</span>
          <el-tag type="info">基于 Dolos</el-tag>
        </div>
      </template>

      <el-tabs v-model="activeTab">
        <!-- 文件上传标签页 -->
        <el-tab-pane label="文件上传" name="upload">
          <div class="upload-section">
            <el-alert
              title="使用说明"
              type="info"
              :closable="false"
              style="margin-bottom: 20px"
            >
              <p>1. 支持上传多个代码文件进行相似度检测</p>
              <p>2. 支持的文件类型：.py, .js, .java, .cpp, .c, .ts, .go, .rs, .php, .rb</p>
              <p>3. 单个文件大小不超过 10MB</p>
            </el-alert>

            <el-upload
              ref="uploadRef"
              class="upload-area"
              drag
              multiple
              :auto-upload="false"
              :on-change="handleFileChange"
              :on-remove="handleFileRemove"
              :file-list="fileList"
              :before-upload="beforeUpload"
              accept=".py,.js,.java,.cpp,.c,.ts,.go,.rs,.php,.rb"
            >
              <i class="el-icon-upload"></i>
              <div class="el-upload__text">
                将文件拖到此处，或<em>点击上传</em>
              </div>
              <template #tip>
                <div class="el-upload__tip">
                  支持 Python, JavaScript, Java, C++, TypeScript, Go, Rust, PHP, Ruby 等代码文件
                </div>
              </template>
            </el-upload>

            <div class="upload-actions" style="margin-top: 20px; text-align: center">
              <el-button
                type="primary"
                size="large"
                :loading="analyzing"
                :disabled="fileList.length < 2"
                @click="handleAnalyze"
              >
                开始分析
              </el-button>
              <el-button size="large" @click="handleClearFiles">
                清空文件
              </el-button>
            </div>
          </div>
        </el-tab-pane>

        <!-- 检测结果标签页 -->
        <el-tab-pane label="检测结果" name="result" :disabled="!analysisResult">
          <div v-if="analysisResult" class="result-section">
            <!-- 概览信息 -->
            <el-card class="overview-card" shadow="hover">
              <template #header>
                <span>分析概览</span>
              </template>
              <el-descriptions :column="3" border>
                <el-descriptions-item label="分析ID">
                  {{ analysisResult.analysis_id }}
                </el-descriptions-item>
                <el-descriptions-item label="分析时间">
                  {{ formatTime(analysisResult.timestamp) }}
                </el-descriptions-item>
                <el-descriptions-item label="文件数量">
                  {{ analysisResult.files?.length || 0 }}
                </el-descriptions-item>
                <el-descriptions-item label="检测对数">
                  {{ analysisResult.pairs?.length || 0 }}
                </el-descriptions-item>
                <el-descriptions-item label="最高相似度">
                  <el-tag :type="getSimilarityTagType(maxSimilarity)">
                    {{ (maxSimilarity * 100).toFixed(2) }}%
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="平均相似度">
                  <el-tag :type="getSimilarityTagType(avgSimilarity)">
                    {{ (avgSimilarity * 100).toFixed(2) }}%
                  </el-tag>
                </el-descriptions-item>
              </el-descriptions>
            </el-card>

            <!-- 相似度图表 -->
            <el-card class="chart-card" shadow="hover" style="margin-top: 20px">
              <template #header>
                <span>相似度分布图</span>
              </template>
              <div ref="chartRef" style="width: 100%; height: 400px"></div>
            </el-card>

            <!-- 详细结果列表 -->
            <el-card class="detail-card" shadow="hover" style="margin-top: 20px">
              <template #header>
                <div style="display: flex; justify-content: space-between; align-items: center">
                  <span>详细检测结果</span>
                  <el-input
                    v-model="searchText"
                    placeholder="搜索文件名"
                    style="width: 200px"
                    clearable
                  />
                </div>
              </template>
              <el-table
                :data="filteredPairs"
                stripe
                border
                style="width: 100%"
                max-height="500"
              >
                <el-table-column type="index" label="#" width="50" />
                <el-table-column prop="leftFile" label="文件1" min-width="150">
                  <template #default="{ row }">
                    <el-tooltip :content="row.leftFile" placement="top">
                      <span class="file-name">{{ getFileName(row.leftFile) }}</span>
                    </el-tooltip>
                  </template>
                </el-table-column>
                <el-table-column prop="rightFile" label="文件2" min-width="150">
                  <template #default="{ row }">
                    <el-tooltip :content="row.rightFile" placement="top">
                      <span class="file-name">{{ getFileName(row.rightFile) }}</span>
                    </el-tooltip>
                  </template>
                </el-table-column>
                <el-table-column prop="similarity" label="相似度" width="120" sortable>
                  <template #default="{ row }">
                    <el-progress
                      :percentage="row.similarity * 100"
                      :color="getProgressColor(row.similarity)"
                      :stroke-width="15"
                    />
                  </template>
                </el-table-column>
                <el-table-column prop="overlap" label="重叠片段数" width="120" />
                <el-table-column label="操作" width="100" fixed="right">
                  <template #default="{ row }">
                    <el-button
                      type="primary"
                      size="small"
                      link
                      @click="handleViewDetail(row)"
                    >
                      查看详情
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>
          </div>
          <el-empty v-else description="暂无分析结果" />
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 详情对话框 -->
    <el-dialog
      v-model="detailDialogVisible"
      title="相似度详情"
      width="80%"
      :close-on-click-modal="false"
    >
      <div v-if="currentDetail" class="detail-content">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="文件1">
            {{ currentDetail.leftFile }}
          </el-descriptions-item>
          <el-descriptions-item label="文件2">
            {{ currentDetail.rightFile }}
          </el-descriptions-item>
          <el-descriptions-item label="相似度">
            {{ (currentDetail.similarity * 100).toFixed(2) }}%
          </el-descriptions-item>
          <el-descriptions-item label="重叠片段数">
            {{ currentDetail.overlap }}
          </el-descriptions-item>
        </el-descriptions>

        <div style="margin-top: 20px">
          <h4>相似代码片段：</h4>
          <el-collapse v-if="currentDetail.fragments && currentDetail.fragments.length > 0">
            <el-collapse-item
              v-for="(fragment, index) in currentDetail.fragments"
              :key="index"
              :title="`片段 ${index + 1}`"
            >
              <el-card>
                <pre class="code-block">{{ fragment }}</pre>
              </el-card>
            </el-collapse-item>
          </el-collapse>
          <el-empty v-else description="无相似片段数据" />
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import { analyzeCode } from '@/api/dolos'

// 数据定义
const activeTab = ref('upload')
const fileList = ref([])
const analyzing = ref(false)
const analysisResult = ref(null)
const uploadRef = ref(null)
const chartRef = ref(null)
const searchText = ref('')
const detailDialogVisible = ref(false)
const currentDetail = ref(null)

let chartInstance = null

// 计算属性
const maxSimilarity = computed(() => {
  if (!analysisResult.value?.pairs) return 0
  return Math.max(...analysisResult.value.pairs.map(p => p.similarity || 0))
})

const avgSimilarity = computed(() => {
  if (!analysisResult.value?.pairs || analysisResult.value.pairs.length === 0) return 0
  const sum = analysisResult.value.pairs.reduce((acc, p) => acc + (p.similarity || 0), 0)
  return sum / analysisResult.value.pairs.length
})

const filteredPairs = computed(() => {
  if (!analysisResult.value?.pairs) return []
  if (!searchText.value) return analysisResult.value.pairs
  
  const search = searchText.value.toLowerCase()
  return analysisResult.value.pairs.filter(pair => 
    pair.leftFile?.toLowerCase().includes(search) ||
    pair.rightFile?.toLowerCase().includes(search)
  )
})

// 方法
const beforeUpload = (file) => {
  const allowedExtensions = ['.py', '.js', '.java', '.cpp', '.c', '.ts', '.go', '.rs', '.php', '.rb']
  const ext = '.' + file.name.split('.').pop().toLowerCase()
  
  if (!allowedExtensions.includes(ext)) {
    ElMessage.error(`不支持的文件类型: ${ext}`)
    return false
  }
  
  const maxSize = 10 * 1024 * 1024 // 10MB
  if (file.size > maxSize) {
    ElMessage.error('文件大小不能超过 10MB')
    return false
  }
  
  return true
}

const handleFileChange = (file, files) => {
  console.log('文件变化:', file, files)
  fileList.value = files
}

const handleFileRemove = (file, files) => {
  console.log('文件移除:', file, files)
  fileList.value = files
}

const handleClearFiles = () => {
  fileList.value = []
  uploadRef.value?.clearFiles()
}

const handleAnalyze = async () => {
  console.log('=== 开始分析 ===')
  console.log('文件列表:', fileList.value)
  console.log('文件数量:', fileList.value.length)

  if (fileList.value.length < 2) {
    ElMessage.warning('请至少上传2个文件进行分析')
    return
  }

  try {
    console.log('开始分析...')
    analyzing.value = true

    const formData = new FormData()
    fileList.value.forEach((file, index) => {
      console.log(`处理文件 ${index}:`, file.name, file)
      const fileData = file.raw || file
      formData.append('files', fileData)
    })

    console.log('发送API请求...')
    const result = await analyzeCode(formData)
    console.log('API返回结果:', result)

    analysisResult.value = result

    ElMessage.success('分析完成')
    activeTab.value = 'result'

    // 渲染图表
    await nextTick()
    renderChart()

  } catch (error) {
    console.error('分析过程出错:', error)
    ElMessage.error(error.message || '分析失败')
  } finally {
    analyzing.value = false
    console.log('=== 分析结束 ===')
  }
}

const renderChart = () => {
  if (!chartRef.value || !analysisResult.value?.pairs) return
  
  if (chartInstance) {
    chartInstance.dispose()
  }
  
  chartInstance = echarts.init(chartRef.value)
  
  const pairs = analysisResult.value.pairs
  const data = pairs.map((p, index) => ({
    name: `${getFileName(p.leftFile)} vs ${getFileName(p.rightFile)}`,
    value: (p.similarity * 100).toFixed(2),
    itemStyle: {
      color: getProgressColor(p.similarity)
    }
  }))
  
  const option = {
    title: {
      text: '文件相似度分布',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      },
      formatter: (params) => {
        const item = params[0]
        return `${item.name}<br/>相似度: ${item.value}%`
      }
    },
    xAxis: {
      type: 'category',
      data: data.map(d => d.name),
      axisLabel: {
        rotate: 45,
        interval: 0,
        fontSize: 10
      }
    },
    yAxis: {
      type: 'value',
      name: '相似度 (%)',
      max: 100
    },
    series: [{
      data: data,
      type: 'bar',
      barWidth: '60%',
      label: {
        show: true,
        position: 'top',
        formatter: '{c}%'
      }
    }]
  }
  
  chartInstance.setOption(option)
}

const getFileName = (path) => {
  if (!path) return ''
  return path.split('/').pop().split('\\').pop()
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  return new Date(timestamp).toLocaleString('zh-CN')
}

const getSimilarityTagType = (similarity) => {
  if (similarity >= 0.7) return 'danger'
  if (similarity >= 0.4) return 'warning'
  return 'success'
}

const getProgressColor = (similarity) => {
  if (similarity >= 0.7) return '#f56c6c'
  if (similarity >= 0.4) return '#e6a23c'
  return '#67c23a'
}

const handleViewDetail = (row) => {
  currentDetail.value = row
  detailDialogVisible.value = true
}
</script>

<style scoped>
.dolos-analysis-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.title {
  font-size: 18px;
  font-weight: bold;
}

.upload-section {
  padding: 20px;
}

.upload-area {
  width: 100%;
}

.file-name {
  display: inline-block;
  max-width: 150px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.code-block {
  background-color: #f5f5f5;
  padding: 10px;
  border-radius: 4px;
  overflow-x: auto;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.5;
}

:deep(.el-upload-dragger) {
  width: 100%;
}
</style>
