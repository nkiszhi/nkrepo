<template>
  <div class="app-container">
    <el-card class="box-card" shadow="never">
      <div slot="header" class="clearfix">
        <span class="card-title">
          <i class="el-icon-document" />
          ATT&CK API组件映射
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
          placeholder="搜索Hash ID、API组件、技术编号等"
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

        <el-table-column label="Hash ID" min-width="150">
          <template #default="scope">
            <el-tooltip effect="dark" :content="scope.row.hash_id" placement="top">
              <span class="hash-id">{{ formatHashId(scope.row.hash_id) }}</span>
            </el-tooltip>
          </template>
        </el-table-column>

        <el-table-column label="API Component" prop="api_component" min-width="150">
          <template #default="scope">
            <el-tag type="info" size="small">{{ scope.row.api_component }}</el-tag>
          </template>
        </el-table-column>

        <el-table-column label="Root Function" prop="root_function" min-width="200">
          <template #default="scope">
            <span>{{ scope.row.root_function || '-' }}</span>
          </template>
        </el-table-column>

        <el-table-column label="包含的技术编号" min-width="250">
          <template #default="scope">
            <div class="technique-tags">
              <el-tag
                v-for="technique in scope.row.technique_ids"
                :key="technique"
                type="danger"
                size="small"
                style="margin: 2px;"
              >
                {{ technique }}
              </el-tag>
              <span v-if="!scope.row.technique_ids || scope.row.technique_ids.length === 0">
                -
              </span>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="创建时间" width="180" align="center">
          <template #default="scope">
            <span>{{ formatDate(scope.row.created_at) }}</span>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="120" align="center">
          <template #default="scope">
            <el-button
              type="text"
              size="small"
              icon="el-icon-view"
              @click="handleDetail(scope.row)"
            >
              详情
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

    <!-- 详情对话框 -->
    <el-dialog
      :title="detailDialog.title"
      v-model="detailDialog.visible"
      width="70%"
      top="5vh"
    >
      <div v-if="detailData" class="detail-container">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="ID">{{ detailData.id }}</el-descriptions-item>
          <el-descriptions-item label="Hash ID">
            <el-tag type="info">{{ detailData.hash_id }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="API Component">
            <el-tag type="primary">{{ detailData.api_component }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Root Function">
            {{ detailData.root_function }}
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ formatDate(detailData.created_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="更新时间">
            {{ formatDate(detailData.updated_at) }}
          </el-descriptions-item>
        </el-descriptions>

        <!-- 技术映射 -->
        <el-card class="detail-section" shadow="never">
          <div slot="header" class="clearfix">
            <span>技术映射</span>
            <el-badge :value="detailData.technique_count || 0" class="item">
              <el-button size="small" type="text">总数</el-button>
            </el-badge>
          </div>
          <div v-if="detailData.techniques && detailData.techniques.length > 0">
            <el-table :data="detailData.techniques" border size="small">
              <el-table-column label="技术编号" prop="technique_id" width="120" />
              <el-table-column label="技术名称" prop="technique_name" />
              <el-table-column label="置信度" prop="confidence" width="100" />
              <el-table-column label="战术ID" prop="tactic_id" width="120" />
              <el-table-column label="战术名称" prop="tactic_name" />
            </el-table>
          </div>
          <div v-else style="text-align: center; color: #999; padding: 20px;">
            暂无技术映射数据
          </div>
        </el-card>

        <!-- 子函数别名 -->
        <el-card v-if="detailData.children_aliases && Object.keys(detailData.children_aliases).length > 0" class="detail-section" shadow="never">
          <div slot="header" class="clearfix">
            <span>子函数别名</span>
            <el-badge :value="detailData.children_aliases_count || 0" class="item">
              <el-button size="small" type="text">总数</el-button>
            </el-badge>
          </div>
          <el-table :data="Object.entries(detailData.children_aliases)" border size="small">
            <el-table-column label="键" prop="0" />
            <el-table-column label="值" prop="1" />
          </el-table>
        </el-card>

        <!-- 其他信息 -->
        <el-collapse v-model="activeCollapse" class="detail-section">
          <el-collapse-item title="代码信息" name="code">
            <el-descriptions :column="1" border>
              <el-descriptions-item label="尝试次数">{{ detailData.tries }}</el-descriptions-item>
              <el-descriptions-item label="生成CPP代码">
                <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 300px; overflow: auto;">
{{ detailData.generated_cpp }}
                </pre>
              </el-descriptions-item>
              <el-descriptions-item label="HLIL源码">
                <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 300px; overflow: auto;">
{{ detailData.source_hlil }}
                </pre>
              </el-descriptions-item>
            </el-descriptions>
          </el-collapse-item>
        </el-collapse>
      </div>

      <div slot="footer" class="dialog-footer">
        <el-button @click="detailDialog.visible = false">关闭</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import attckApi from '@/api/attck'

export default {
  name: 'ApiComponentMapping',
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
      detailDialog: {
        visible: false,
        title: 'API组件详情'
      },
      detailData: null,
      activeCollapse: []
    }
  },
  created() {
    this.fetchData()
  },
  methods: {
    // 获取数据
    async fetchData() {
      this.listLoading = true
      try {
        console.log('🔄 开始获取API组件数据...')
        console.log('请求参数:', {
          page: this.listQuery.page,
          pageSize: this.listQuery.pageSize,
          search: this.searchQuery || undefined
        })

        const response = await attckApi.getApiComponents({
          page: this.listQuery.page,
          pageSize: this.listQuery.pageSize,
          search: this.searchQuery || undefined
        })

        console.log('API响应:', response)

        if (response.success) {
          this.list = response.data
          this.total = response.pagination.total
          console.log(`✅ 获取API组件数据成功，共${response.pagination.total}条记录`)
        } else {
          console.error('API返回错误:', response.error)
          this.$message.error(response.error || '获取数据失败')
        }
      } catch (error) {
        console.error('获取API组件数据失败:', error)
        // 检查具体的错误信息
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

    // 查看详情
    async handleDetail(row) {
      try {
        console.log('查看详情:', row.hash_id, row.api_component)
        const response = await attckApi.getApiComponentDetail(row.hash_id, row.api_component)

        console.log('详情响应:', response)

        if (response.success) {
          this.detailData = response.data
          this.detailDialog.title = `API组件详情 - ${row.api_component}`
          this.detailDialog.visible = true
        } else {
          console.error('获取详情失败:', response.error)
          this.$message.error(response.error || '获取详情失败')
        }
      } catch (error) {
        console.error('获取详情失败:', error)
        if (error.response) {
          console.error('响应状态码:', error.response.status)
          console.error('响应数据:', error.response.data)
          this.$message.error(`获取详情失败: ${error.response.status}`)
        } else {
          this.$message.error('获取详情失败')
        }
      }
    },

    // 格式化Hash ID（显示前8位...后8位）
    formatHashId(hashId) {
      if (!hashId || hashId.length <= 16) return hashId
      return `${hashId.substring(0, 8)}...${hashId.substring(hashId.length - 8)}`
    },

    // 格式化日期
    formatDate(dateStr) {
      if (!dateStr) return '-'
      try {
        const date = new Date(dateStr)
        if (isNaN(date.getTime())) return dateStr
        return date.toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        })
      } catch (e) {
        return dateStr
      }
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

.hash-id {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #666;
  cursor: pointer;
}

.technique-tags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}

.detail-container {
  max-height: 70vh;
  overflow-y: auto;
}

.detail-section {
  margin-top: 20px;
}

.el-descriptions {
  margin-bottom: 20px;
}

.el-badge {
  margin-left: 10px;
}

pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.4;
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
}
</style>
