<template>
  <div class="app-container">
    <el-card class="box-card" shadow="never">
      <div slot="header" class="clearfix">
        <span class="card-title">
          <i class="el-icon-collection" />
          ATT&CK技术映射管理
        </span>
        <div class="card-header-right">
          <el-button type="primary" icon="el-icon-refresh" size="small" @click="refreshTable">
            刷新
          </el-button>
        </div>
      </div>

      <!-- 搜索栏 -->
      <div class="filter-container">
        <el-input
          v-model="searchQuery"
          placeholder="搜索技术编号、技术名称、战术名称等"
          style="width: 300px; margin-right: 10px;"
          clearable
          @keyup.enter.native="handleSearch"
        />
        <el-button type="primary" icon="el-icon-search" size="small" @click="handleSearch">
          搜索
        </el-button>
        <el-button size="small" @click="resetSearch">重置</el-button>
      </div>

      <!-- 数据表格 -->
      <el-table
        v-loading="listLoading"
        :data="list"
        border
        fit
        highlight-current-row
        style="width: 100%; margin-top: 20px;"
      >
        <el-table-column label="ID" prop="id" align="center" width="80">
          <template #default="scope">
            <span>{{ scope.row.id }}</span>
          </template>
        </el-table-column>

        <el-table-column label="技术编号" prop="technique_id" min-width="120" align="center">
          <template #default="scope">
            <el-tag type="danger" size="small">{{ scope.row.technique_id }}</el-tag>
          </template>
        </el-table-column>

        <el-table-column label="技术名称" prop="technique_name" min-width="200">
          <template #default="scope">
            <span>{{ scope.row.technique_name || '-' }}</span>
          </template>
        </el-table-column>

        <el-table-column label="战术名称" prop="tactic_name" min-width="150">
          <template #default="scope">
            <el-tag type="warning" size="small">{{ scope.row.tactic_name || '-' }}</el-tag>
          </template>
        </el-table-column>

        <el-table-column label="关联函数数量" prop="function_count" width="120" align="center">
          <template #default="scope">
            <!-- 直接显示数字，不加"函数"字样 -->
            <el-tag :type="getCountType(scope.row.function_count)" size="small">
              {{ scope.row.function_count || 0 }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="120" align="center">
          <template #default="scope">
            <el-button
              type="text"
              size="small"
              icon="el-icon-view"
              :disabled="!scope.row.function_count || scope.row.function_count === 0"
              @click="handleViewFunctions(scope.row)"
            >
              查看
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-container" style="margin-top: 20px;">
        <el-pagination
          v-show="total > 0"
          :current-page="listQuery.page"
          :page-sizes="[10, 20, 30, 50]"
          :page-size="listQuery.pageSize"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 函数详情对话框 -->
    <el-dialog
      :title="functionDialog.title"
      v-model="functionDialog.visible"
      width="60%"
      top="5vh"
    >
      <div v-if="functionData" class="detail-container">
        <div class="function-header">
          <el-descriptions :column="3" border>
            <el-descriptions-item label="技术编号">
              <el-tag type="danger">{{ functionData.technique_id }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="关联函数总数">
              <el-tag type="primary">{{ functionData.total || 0 }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="查询时间">
              {{ formatDate(new Date()) }}
            </el-descriptions-item>
          </el-descriptions>
        </div>

        <!-- 函数列表 -->
        <el-card class="function-section" shadow="never" style="margin-top: 20px;">
          <div slot="header" class="clearfix">
            <span>关联函数列表</span>
            <span style="float: right; font-size: 12px; color: #909399;">
              共 {{ functionData.total }} 个函数
            </span>
          </div>
          <el-table
            :data="functionData.functions"
            border
            size="small"
            style="width: 100%"
          >
            <el-table-column label="序号" prop="id" width="80" align="center" />
            <el-table-column label="函数名" prop="function_name" min-width="200">
              <template #default="scope">
                <el-tag type="primary" size="small">{{ scope.row.function_name }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="文件名称" prop="file_name" min-width="250">
              <template #default="scope">
                <el-tooltip effect="dark" :content="scope.row.file_name" placement="top">
                  <span class="file-name">{{ formatFileName(scope.row.file_name) }}</span>
                </el-tooltip>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </div>

      <div slot="footer" class="dialog-footer">
        <el-button @click="functionDialog.visible = false">关闭</el-button>
        <el-button
          type="primary"
          :disabled="!functionData || !functionData.functions || functionData.functions.length === 0"
          @click="exportFunctions"
        >
          导出CSV
        </el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import attckApi from '@/api/attck'
import { exportToCSV } from '@/utils/export'

export default {
  name: 'TechniqueMapping',
  data() {
    return {
      list: [],
      total: 0,
      listLoading: false,
      searchQuery: '',
      listQuery: {
        page: 1,
        pageSize: 20
      },
      functionDialog: {
        visible: false,
        title: '关联函数列表'
      },
      functionData: null
    }
  },
  created() {
    this.fetchData()
  },
  methods: {
    // 获取技术映射数据
    async fetchData() {
      this.listLoading = true
      try {
        console.log('🔄 开始获取技术映射数据...')
        console.log('请求参数:', {
          page: this.listQuery.page,
          pageSize: this.listQuery.pageSize,
          search: this.searchQuery || undefined
        })

        const response = await attckApi.getTechniqueMapping({
          page: this.listQuery.page,
          pageSize: this.listQuery.pageSize,
          search: this.searchQuery || undefined
        })

        console.log('技术映射响应:', response)

        if (response.success) {
          this.list = response.data
          this.total = response.pagination.total
          console.log(`✅ 获取技术映射数据成功，共${response.pagination.total}条记录`)
        } else {
          console.error('API返回错误:', response.error)
          this.$message.error(response.error || '获取数据失败')
        }
      } catch (error) {
        console.error('获取技术映射数据失败:', error)
        if (error.response) {
          console.error('响应状态码:', error.response.status)
          console.error('响应数据:', error.response.data)
          this.$message.error(`请求失败: ${error.response.status} ${error.response.statusText}`)
        } else if (error.request) {
          console.error('无响应:', error.request)
          this.$message.error('服务器无响应，请检查网络连接')
        } else {
          console.error('请求错误:', error.message)
          this.$message.error(`请求错误: ${error.message}`)
        }
      } finally {
        this.listLoading = false
      }
    },

    // 根据数量返回不同的标签类型
    getCountType(count) {
      if (!count || count === 0) return 'info'
      if (count >= 10) return 'success'
      if (count >= 5) return 'warning'
      return 'primary'
    },

    // 处理搜索
    handleSearch() {
      this.listQuery.page = 1
      this.fetchData()
    },

    // 重置搜索
    resetSearch() {
      this.searchQuery = ''
      this.listQuery.page = 1
      this.fetchData()
    },

    // 刷新表格
    refreshTable() {
      this.fetchData()
      this.$message.success('数据已刷新')
    },

    // 分页大小改变
    handleSizeChange(val) {
      this.listQuery.pageSize = val
      this.fetchData()
    },

    // 当前页改变
    handleCurrentChange(val) {
      this.listQuery.page = val
      this.fetchData()
    },

    // 查看函数详情
    async handleViewFunctions(row) {
      try {
        console.log('查看函数详情:', row.technique_id)
        this.listLoading = true

        const response = await attckApi.getTechniqueFunctions(row.technique_id)

        console.log('函数详情响应:', response)

        if (response.success) {
          this.functionData = {
            technique_id: row.technique_id,
            technique_name: row.technique_name,
            tactic_name: row.tactic_name,
            functions: response.data,
            total: response.total
          }
          this.functionDialog.title = `${row.technique_id} - 关联函数列表 (${response.total}个)`
          this.functionDialog.visible = true
        } else {
          console.error('获取函数详情失败:', response.error)
          this.$message.error(response.error || '获取函数详情失败')
        }
      } catch (error) {
        console.error('获取函数详情失败:', error)
        if (error.response) {
          console.error('响应状态码:', error.response.status)
          console.error('响应数据:', error.response.data)
          this.$message.error(`获取函数详情失败: ${error.response.status}`)
        } else {
          this.$message.error('获取函数详情失败')
        }
      } finally {
        this.listLoading = false
      }
    },

    // 格式化文件名（显示前8位...后8位）
    formatFileName(fileName) {
      if (!fileName || fileName.length <= 16) return fileName
      return `${fileName.substring(0, 8)}...${fileName.substring(fileName.length - 8)}`
    },

    // 格式化日期
    formatDate(date) {
      if (!date) return '-'
      try {
        const dateObj = date instanceof Date ? date : new Date(date)
        if (isNaN(dateObj.getTime())) return '-'
        return dateObj.toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        })
      } catch (e) {
        return '-'
      }
    },

    // 导出函数列表为CSV
    exportFunctions() {
      if (!this.functionData || !this.functionData.functions) {
        this.$message.warning('没有数据可以导出')
        return
      }

      const data = this.functionData.functions.map(item => ({
        '序号': item.id,
        '技术编号': this.functionData.technique_id,
        '技术名称': this.functionData.technique_name,
        '战术名称': this.functionData.tactic_name,
        '函数名': item.function_name,
        '文件名称': item.file_name
      }))

      const filename = `技术映射_${this.functionData.technique_id}_${new Date().getTime()}.csv`

      exportToCSV(data, filename)
      this.$message.success('导出成功')
    }
  }
}
</script>

<style scoped>
.app-container {
  padding: 20px;
}

.card-title {
  font-size: 18px;
  font-weight: bold;
  color: #333;
}

.card-title i {
  margin-right: 8px;
  color: #409EFF;
}

.card-header-right {
  float: right;
}

.filter-container {
  margin-bottom: 20px;
}

.file-name {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #666;
  cursor: pointer;
}

.detail-container {
  max-height: 70vh;
  overflow-y: auto;
}

.function-section {
  margin-top: 20px;
}

.function-header {
  margin-bottom: 15px;
}

/* 添加响应式设计 */
@media screen and (max-width: 768px) {
  .app-container {
    padding: 10px;
  }

  .filter-container {
    display: flex;
    flex-direction: column;
  }

  .filter-container .el-input {
    width: 100%;
    margin-right: 0;
    margin-bottom: 10px;
  }

  .card-header-right {
    float: none;
    margin-top: 10px;
  }

  .el-dialog {
    width: 95% !important;
  }
}
</style>
