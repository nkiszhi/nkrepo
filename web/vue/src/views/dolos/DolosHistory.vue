<template>
  <div class="dolos-history-container">
    <el-card class="main-card">
      <template #header>
        <div class="card-header">
          <span class="title">检测历史记录</span>
          <el-button type="primary" @click="handleRefresh">
            <el-icon><refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </template>

      <!-- 搜索和筛选 -->
      <div class="filter-section" style="margin-bottom: 20px">
        <el-row :gutter="20">
          <el-col :span="8">
            <el-input
              v-model="searchText"
              placeholder="搜索文件名或分析ID"
              clearable
              @clear="handleSearch"
              @keyup.enter="handleSearch"
            >
              <template #prefix>
                <el-icon><search /></el-icon>
              </template>
            </el-input>
          </el-col>
          <el-col :span="6">
            <el-date-picker
              v-model="dateRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              value-format="YYYY-MM-DD"
              @change="handleSearch"
            />
          </el-col>
          <el-col :span="4">
            <el-button type="primary" @click="handleSearch">
              <el-icon><search /></el-icon>
              搜索
            </el-button>
          </el-col>
        </el-row>
      </div>

      <!-- 历史记录表格 -->
      <el-table
        v-loading="loading"
        :data="historyData"
        stripe
        border
        style="width: 100%"
      >
        <el-table-column type="index" label="#" width="50" />
        <el-table-column prop="analysis_id" label="分析ID" width="280">
          <template #default="{ row }">
            <el-tooltip :content="row.analysis_id" placement="top">
              <span class="id-text">{{ row.analysis_id.substring(0, 8) }}...</span>
            </el-tooltip>
          </template>
        </el-table-column>
        <el-table-column prop="timestamp" label="分析时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.timestamp) }}
          </template>
        </el-table-column>
        <el-table-column prop="files" label="文件列表" min-width="200">
          <template #default="{ row }">
            <el-tag
              v-for="(file, index) in row.files.slice(0, 3)"
              :key="index"
              style="margin-right: 5px; margin-bottom: 5px"
              size="small"
            >
              {{ file }}
            </el-tag>
            <el-tag
              v-if="row.files.length > 3"
              type="info"
              size="small"
            >
              +{{ row.files.length - 3 }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="file_count" label="文件数" width="80">
          <template #default="{ row }">
            {{ row.files?.length || 0 }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button
              type="primary"
              size="small"
              link
              @click="handleViewResult(row)"
            >
              <el-icon><view /></el-icon>
              查看结果
            </el-button>
            <el-button
              type="danger"
              size="small"
              link
              @click="handleDelete(row)"
            >
              <el-icon><delete /></el-icon>
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-section" style="margin-top: 20px; text-align: right">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 结果详情对话框 -->
    <el-dialog
      v-model="resultDialogVisible"
      title="分析结果详情"
      width="90%"
      :close-on-click-modal="false"
    >
      <div v-if="currentResult" class="result-detail">
        <el-card shadow="hover" style="margin-bottom: 20px">
          <template #header>
            <span>基本信息</span>
          </template>
          <el-descriptions :column="3" border>
            <el-descriptions-item label="分析ID">
              {{ currentResult.analysis_id }}
            </el-descriptions-item>
            <el-descriptions-item label="分析时间">
              {{ formatTime(currentResult.timestamp) }}
            </el-descriptions-item>
            <el-descriptions-item label="文件数量">
              {{ currentResult.files?.length || 0 }}
            </el-descriptions-item>
          </el-descriptions>
        </el-card>

        <el-card shadow="hover">
          <template #header>
            <span>相似度分析结果</span>
          </template>
          <el-table
            :data="currentResult.pairs"
            stripe
            border
            max-height="400"
          >
            <el-table-column type="index" label="#" width="50" />
            <el-table-column prop="leftFile" label="文件1" min-width="150">
              <template #default="{ row }">
                {{ getFileName(row.leftFile) }}
              </template>
            </el-table-column>
            <el-table-column prop="rightFile" label="文件2" min-width="150">
              <template #default="{ row }">
                {{ getFileName(row.rightFile) }}
              </template>
            </el-table-column>
            <el-table-column prop="similarity" label="相似度" width="150">
              <template #default="{ row }">
                <el-progress
                  :percentage="row.similarity * 100"
                  :color="getProgressColor(row.similarity)"
                />
              </template>
            </el-table-column>
            <el-table-column prop="overlap" label="重叠片段数" width="120" />
          </el-table>
        </el-card>
      </div>
      <el-empty v-else description="无结果数据" />
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { 
  Refresh, 
  Search, 
  View, 
  Delete 
} from '@element-plus/icons-vue'
import { getAnalysisHistory, getAnalysisResult, deleteAnalysisResult } from '@/api/dolos'

// 数据定义
const loading = ref(false)
const historyData = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const searchText = ref('')
const dateRange = ref([])
const resultDialogVisible = ref(false)
const currentResult = ref(null)

// 方法
const fetchHistory = async () => {
  try {
    loading.value = true
    const params = {
      skip: (currentPage.value - 1) * pageSize.value,
      limit: pageSize.value,
      search: searchText.value,
      start_date: dateRange.value?.[0],
      end_date: dateRange.value?.[1]
    }
    
    const response = await getAnalysisHistory(params)
    historyData.value = response.items || response
    total.value = response.total || historyData.value.length
  } catch (error) {
    ElMessage.error(error.message || '获取历史记录失败')
  } finally {
    loading.value = false
  }
}

const handleRefresh = () => {
  fetchHistory()
}

const handleSearch = () => {
  currentPage.value = 1
  fetchHistory()
}

const handleSizeChange = (val) => {
  pageSize.value = val
  fetchHistory()
}

const handleCurrentChange = (val) => {
  currentPage.value = val
  fetchHistory()
}

const handleViewResult = async (row) => {
  try {
    const result = await getAnalysisResult(row.analysis_id)
    currentResult.value = result
    resultDialogVisible.value = true
  } catch (error) {
    ElMessage.error(error.message || '获取结果详情失败')
  }
}

const handleDelete = async (row) => {
  try {
    await ElMessageBox.confirm(
      '确定要删除这条分析记录吗？',
      '确认删除',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    await deleteAnalysisResult(row.analysis_id)
    ElMessage.success('删除成功')
    fetchHistory()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(error.message || '删除失败')
    }
  }
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  return new Date(timestamp).toLocaleString('zh-CN')
}

const getFileName = (path) => {
  if (!path) return ''
  return path.split('/').pop().split('\\').pop()
}

const getProgressColor = (similarity) => {
  if (similarity >= 0.7) return '#f56c6c'
  if (similarity >= 0.4) return '#e6a23c'
  return '#67c23a'
}

// 生命周期
onMounted(() => {
  fetchHistory()
})
</script>

<style scoped>
.dolos-history-container {
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

.id-text {
  font-family: 'Courier New', monospace;
  font-size: 13px;
}
</style>
