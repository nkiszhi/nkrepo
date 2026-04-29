<template>
  <div class="av-scan-history-container">
    <!-- 标题 -->
    <div class="text-center">
      <h2 class="text-primary">检测历史记录</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
    </div>

    <!-- 筛选区域 -->
    <div class="filter-section">
      <el-form :inline="true" :model="filters" class="filter-form">
        <el-form-item label="时间范围">
          <el-date-picker
            v-model="filters.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            value-format="YYYY-MM-DD"
          />
        </el-form-item>
        <el-form-item label="检测结果">
          <el-select v-model="filters.resultType" placeholder="全部" clearable>
            <el-option label="全部" value="" />
            <el-option label="有恶意" value="malicious" />
            <el-option label="全部安全" value="safe" />
          </el-select>
        </el-form-item>
        <el-form-item label="文件名">
          <el-input v-model="filters.fileName" placeholder="搜索文件名" clearable />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="fetchHistoryList">查询</el-button>
          <el-button @click="resetFilters">重置</el-button>
        </el-form-item>
      </el-form>
    </div>

    <!-- 历史记录列表 -->
    <div class="history-list-section">
      <el-table :data="historyList" border style="width: 100%" v-loading="loading">
        <el-table-column type="index" label="序号" width="60" />
        <el-table-column prop="task_id" label="任务ID" width="280" />
        <el-table-column prop="total_files" label="文件数" width="80" />
        <el-table-column label="引擎数" width="80">
          <template #default="scope">
            {{ scope.row.selected_engines ? scope.row.selected_engines.length : 0 }}
          </template>
        </el-table-column>
        <el-table-column label="检测结果" width="150">
          <template #default="scope">
            <el-tag type="danger" v-if="scope.row.malicious_count > 0">
              恶意: {{ scope.row.malicious_count }}
            </el-tag>
            <el-tag type="success" v-else>全部安全</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">
              {{ getStatusText(scope.row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="180">
          <template #default="scope">
            {{ formatDateTime(scope.row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150" fixed="right">
          <template #default="scope">
            <el-button type="primary" size="small" link @click="viewDetail(scope.row.task_id)">
              查看详情
            </el-button>
            <el-button type="danger" size="small" link @click="deleteHistory(scope.row.task_id)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="pagination.total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="fetchHistoryList"
          @current-change="fetchHistoryList"
        />
      </div>
    </div>

    <!-- 详情弹窗 -->
    <el-dialog v-model="detailDialogVisible" title="检测详情" width="80%" top="5vh">
      <div v-if="currentDetail" class="detail-content">
        <!-- 任务信息 -->
        <div class="detail-header">
          <div class="detail-item">
            <span class="label">任务ID:</span>
            <span class="value">{{ currentDetail.task_id }}</span>
          </div>
          <div class="detail-item">
            <span class="label">文件数:</span>
            <span class="value">{{ currentDetail.total_files }}</span>
          </div>
          <div class="detail-item">
            <span class="label">使用引擎:</span>
            <span class="value">{{ currentDetail.selected_engines ? currentDetail.selected_engines.join(', ') : '' }}</span>
          </div>
          <div class="detail-item">
            <span class="label">创建时间:</span>
            <span class="value">{{ formatDateTime(currentDetail.created_at) }}</span>
          </div>
        </div>

        <!-- 结果表格 -->
        <el-table :data="currentDetail.results" border style="width: 100%" max-height="500">
          <el-table-column prop="file_name" label="文件名" fixed width="200" />
          <el-table-column
            v-for="engine in currentDetail.selected_engines"
            :key="engine"
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
          <el-table-column label="标签" width="150" fixed="right">
            <template #default="scope">
              <div class="tag-cell">
                <el-tag v-if="scope.row.tag" :type="getTagType(scope.row.tag_type)" size="small">
                  {{ scope.row.tag }}
                </el-tag>
                <el-button type="text" size="small" @click="openTagDialog(scope.row)">
                  {{ scope.row.tag ? '编辑' : '添加' }}标签
                </el-button>
              </div>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-dialog>

    <!-- 标签编辑弹窗 -->
    <el-dialog v-model="tagDialogVisible" title="添加/编辑标签" width="400px">
      <el-form :model="tagForm" label-width="80px">
        <el-form-item label="预定义标签">
          <el-radio-group v-model="tagForm.tag">
            <el-radio label="确认恶意">确认恶意</el-radio>
            <el-radio label="误报">误报</el-radio>
            <el-radio label="安全">安全</el-radio>
            <el-radio label="待确认">待确认</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="自定义标签">
          <el-input v-model="tagForm.customTag" placeholder="输入自定义标签" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="tagDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveTag">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import axios from 'axios'
import { ElMessageBox, ElMessage } from 'element-plus'

const apiService = axios.create({
  timeout: 60000,
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
  name: 'AVScanHistory',
  data() {
    return {
      filters: {
        dateRange: [],
        resultType: '',
        fileName: ''
      },
      pagination: {
        page: 1,
        pageSize: 20,
        total: 0
      },
      historyList: [],
      loading: false,
      apiBaseUrl: '',
      detailDialogVisible: false,
      currentDetail: null,
      tagDialogVisible: false,
      tagForm: {
        taskId: '',
        fileName: '',
        tag: '',
        customTag: '',
        tagType: 'predefined'
      }
    }
  },
  created() {
    this.loadConfig()
    this.fetchHistoryList()
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

    async fetchHistoryList() {
      this.loading = true
      try {
        const params = {
          page: this.pagination.page,
          page_size: this.pagination.pageSize
        }
        if (this.filters.dateRange && this.filters.dateRange.length === 2) {
          params.start_date = this.filters.dateRange[0]
          params.end_date = this.filters.dateRange[1]
        }
        if (this.filters.resultType) {
          params.result_type = this.filters.resultType
        }
        if (this.filters.fileName) {
          params.file_name = this.filters.fileName
        }

        const response = await apiService.get(`${this.apiBaseUrl}/api/av_history/list`, { params })
        this.historyList = response.data.items
        this.pagination.total = response.data.total
      } catch (error) {
        console.error('获取历史记录失败:', error)
        this.$message.error('获取历史记录失败')
      } finally {
        this.loading = false
      }
    },

    resetFilters() {
      this.filters = {
        dateRange: [],
        resultType: '',
        fileName: ''
      }
      this.pagination.page = 1
      this.fetchHistoryList()
    },

    async viewDetail(taskId) {
      try {
        const response = await apiService.get(`${this.apiBaseUrl}/api/av_history/detail/${taskId}`)
        this.currentDetail = response.data
        this.detailDialogVisible = true
      } catch (error) {
        console.error('获取详情失败:', error)
        this.$message.error('获取详情失败')
      }
    },

    async deleteHistory(taskId) {
      try {
        await ElMessageBox.confirm(
          '确定要删除这条记录吗？',
          '确认删除',
          {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
          }
        )
        
        await apiService.delete(`${this.apiBaseUrl}/api/av_history/${taskId}`)
        ElMessage.success('删除成功')
        this.fetchHistoryList()
      } catch (error) {
        if (error !== 'cancel') {
          console.error('删除失败:', error)
          ElMessage.error(error.response?.data?.detail || error.message || '删除失败')
        }
      }
    },

    openTagDialog(row) {
      this.tagForm = {
        taskId: this.currentDetail.task_id,
        fileName: row.file_name,
        tag: row.tag || '',
        customTag: row.tag_type === 'custom' ? row.tag : '',
        tagType: row.tag_type || 'predefined'
      }
      this.tagDialogVisible = true
    },

    async saveTag() {
      try {
        const tag = this.tagForm.customTag || this.tagForm.tag
        if (!tag) {
          this.$message.error('请输入或选择标签')
          return
        }

        await apiService.post(`${this.apiBaseUrl}/api/av_history/tag`, {
          task_id: this.tagForm.taskId,
          file_name: this.tagForm.fileName,
          tag: tag,
          tag_type: this.tagForm.customTag ? 'custom' : 'predefined'
        })

        this.$message.success('标签保存成功')
        this.tagDialogVisible = false

        // 刷新详情
        await this.viewDetail(this.tagForm.taskId)
      } catch (error) {
        console.error('保存标签失败:', error)
        this.$message.error('保存标签失败')
      }
    },

    formatDateTime(dateStr) {
      if (!dateStr) return ''
      const date = new Date(dateStr)
      return date.toLocaleString('zh-CN')
    },

    getStatusType(status) {
      const typeMap = { 'pending': 'info', 'running': 'warning', 'completed': 'success', 'failed': 'danger' }
      return typeMap[status] || 'info'
    },

    getStatusText(status) {
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
    },

    getTagType(tagType) {
      return tagType === 'predefined' ? 'primary' : 'info'
    }
  }
}
</script>

<style scoped>
.av-scan-history-container {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.filter-section {
  background: white;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.filter-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.history-list-section {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.detail-content {
  padding: 20px;
}

.detail-header {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
}

.detail-item {
  display: flex;
  gap: 10px;
}

.detail-item .label {
  font-weight: bold;
  color: #606266;
}

.detail-item .value {
  color: #303133;
}

.tag-cell {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.malicious-count {
  color: #f56c6c;
  font-weight: bold;
}
</style>
