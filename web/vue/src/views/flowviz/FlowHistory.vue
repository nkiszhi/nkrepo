<!-- vue/src/views/flowviz/FlowHistory.vue -->
<template>
  <div class="flow-history">
    <el-container>
      <el-header style="height: auto; padding: 20px;">
        <el-page-header content="分析历史记录" @back="$router.go(-1)">
          <template slot="title">
            返回分析页面
          </template>
        </el-page-header>
      </el-header>

      <el-main>
        <el-card>
          <div slot="header">
            <span style="font-size: 18px; font-weight: bold;">历史分析记录</span>
            <el-button
              style="float: right; padding: 3px 0"
              type="text"
              :disabled="savedFlows.length === 0"
              @click="clearHistory"
            >
              清空历史
            </el-button>
          </div>

          <div v-if="savedFlows.length === 0" class="no-history">
            <el-empty description="暂无历史记录">
              <el-button type="primary" @click="$router.push('/flowviz/analysis')">
                开始新的分析
              </el-button>
            </el-empty>
          </div>

          <div v-else class="history-list">
            <el-table :data="paginatedFlows" style="width: 100%">
              <el-table-column prop="title" label="标题" width="180">
                <template #default="scope">
                  <div class="flow-title">
                    <i class="el-icon-document" style="margin-right: 5px;" />
                    {{ scope.row.title }}
                  </div>
                </template>
              </el-table-column>

              <el-table-column prop="inputType" label="输入类型" width="100">
                <template #default="scope">
                  <el-tag :type="scope.row.inputType === 'url' ? 'warning' : 'primary'" size="small">
                    {{ scope.row.inputType === 'url' ? 'URL' : '文本' }}
                  </el-tag>
                </template>
              </el-table-column>

              <el-table-column prop="inputValue" label="输入内容" width="300">
                <template #default="scope">
                  <div class="input-preview">
                    {{ scope.row.inputValue }}
                  </div>
                </template>
              </el-table-column>

              <el-table-column prop="nodes.length" label="节点数" width="80" align="center">
                <template #default="scope">
                  <el-tag size="small">{{ scope.row.nodes.length }}</el-tag>
                </template>
              </el-table-column>

              <el-table-column prop="edges.length" label="边数" width="80" align="center">
                <template #default="scope">
                  <el-tag size="small">{{ scope.row.edges.length }}</el-tag>
                </template>
              </el-table-column>

              <el-table-column prop="analysisTime" label="分析时间" width="100" align="center">
                <template #default="scope">
                  <span>{{ scope.row.analysisTime }}秒</span>
                </template>
              </el-table-column>

              <el-table-column prop="createdAt" label="创建时间" width="180">
                <template #default="scope">
                  {{ formatDate(scope.row.createdAt) }}
                </template>
              </el-table-column>

              <el-table-column label="操作" width="200" fixed="right">
                <template #default="scope">
                  <el-button-group>
                    <el-button
                      size="mini"
                      title="加载此分析"
                      @click="loadFlow(scope.row)"
                    >
                      加载
                    </el-button>
                    <el-button
                      size="mini"
                      type="success"
                      title="导出JSON"
                      @click="exportFlow(scope.row, 'json')"
                    >
                      JSON
                    </el-button>
                    <el-button
                      size="mini"
                      type="danger"
                      title="删除"
                      @click="deleteFlow(scope.row.id)"
                    >
                      删除
                    </el-button>
                  </el-button-group>
                </template>
              </el-table-column>
            </el-table>

            <div class="pagination-container">
              <el-pagination
                :current-page="currentPage"
                :page-sizes="[10, 20, 50, 100]"
                :page-size="pageSize"
                layout="total, sizes, prev, pager, next"
                :total="savedFlows.length"
                @size-change="handleSizeChange"
                @current-change="handleCurrentChange"
              />
            </div>
          </div>
        </el-card>
      </el-main>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'FlowHistory',
  data() {
    return {
      savedFlows: [],
      currentPage: 1,
      pageSize: 10
    }
  },

  computed: {
    paginatedFlows() {
      const start = (this.currentPage - 1) * this.pageSize
      const end = start + this.pageSize
      return this.savedFlows.slice(start, end)
    }
  },

  mounted() {
    this.loadSavedFlows()
  },

  methods: {
    loadSavedFlows() {
      const flows = localStorage.getItem('flowviz_saved_flows')
      this.savedFlows = flows ? JSON.parse(flows) : []
    },

    formatDate(dateStr) {
      const date = new Date(dateStr)
      return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      })
    },

    loadFlow(flow) {
      // 跳转到分析页面并加载数据
      this.$router.push({
        path: '/flowviz/analysis',
        query: {
          loadFlow: flow.id
        }
      })
    },

    exportFlow(flow, format) {
      const dataStr = JSON.stringify(flow, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)

      const link = document.createElement('a')
      link.download = `flow-${flow.id}.json`
      link.href = url
      link.click()

      URL.revokeObjectURL(url)

      this.$message.success('导出成功')
    },

    deleteFlow(id) {
      this.$confirm('确定要删除此分析记录吗?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.savedFlows = this.savedFlows.filter(flow => flow.id !== id)
        localStorage.setItem('flowviz_saved_flows', JSON.stringify(this.savedFlows))
        this.$message.success('删除成功')
      }).catch(() => {})
    },

    clearHistory() {
      this.$confirm('确定要清空所有历史记录吗?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.savedFlows = []
        localStorage.removeItem('flowviz_saved_flows')
        this.$message.success('已清空历史记录')
      }).catch(() => {})
    },

    handleSizeChange(val) {
      this.pageSize = val
      this.currentPage = 1
    },

    handleCurrentChange(val) {
      this.currentPage = val
    }
  }
}
</script>

<style scoped>
.flow-history {
  height: 100%;
  padding: 20px;
}

.no-history {
  padding: 50px 0;
  text-align: center;
}

.flow-title {
  display: flex;
  align-items: center;
  font-weight: 500;
}

.input-preview {
  font-size: 12px;
  color: #606266;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pagination-container {
  margin-top: 20px;
  text-align: center;
}
</style>
