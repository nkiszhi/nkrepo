<!-- vue/src/views/flowviz/FlowHistory.vue -->
<template>
  <div class="flow-history">
    <el-container>
      <el-header style="height: auto; padding: 20px;">
        <el-page-header content="分析历史记录" @back="$router.go(-1)">
          <template #title>
            返回分析页面
          </template>
        </el-page-header>
      </el-header>

      <el-main>
        <el-card>
          <template #header>
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <span style="font-size: 18px; font-weight: bold;">历史分析记录 (共{{ savedFlows.length }}条)</span>
              <el-button
                type="text"
                :disabled="savedFlows.length === 0"
                @click="clearHistory"
              >
                清空历史
              </el-button>
            </div>
          </template>

          <div v-if="savedFlows.length === 0" class="no-history">
            <el-empty description="暂无历史记录">
              <el-button type="primary" @click="$router.push('/flowviz/analysis')">
                开始新的分析
              </el-button>
            </el-empty>
          </div>

          <div v-else class="history-list">
            <el-table :data="savedFlows" border stripe style="width: 100%">
              <el-table-column label="标题" min-width="150">
                <template #default="scope">
                  <div class="flow-title">
                    <i class="el-icon-document" style="margin-right: 5px;" />
                    {{ scope.row.title || '未命名' }}
                  </div>
                </template>
              </el-table-column>

              <el-table-column label="输入类型" width="100">
                <template #default="scope">
                  <el-tag :type="scope.row.inputType === 'url' ? 'warning' : 'primary'" size="small">
                    {{ scope.row.inputType === 'url' ? 'URL' : '文本' }}
                  </el-tag>
                </template>
              </el-table-column>

              <el-table-column label="输入内容" min-width="200" show-overflow-tooltip>
                <template #default="scope">
                  <div class="input-preview">
                    {{ scope.row.inputValue || scope.row.input || '-' }}
                  </div>
                </template>
              </el-table-column>

              <el-table-column label="节点数" width="80" align="center">
                <template #default="scope">
                  <el-tag size="small">{{ (scope.row.nodes && scope.row.nodes.length) || 0 }}</el-tag>
                </template>
              </el-table-column>

              <el-table-column label="边数" width="80" align="center">
                <template #default="scope">
                  <el-tag size="small">{{ (scope.row.edges && scope.row.edges.length) || 0 }}</el-tag>
                </template>
              </el-table-column>

              <el-table-column label="分析时间" width="100" align="center">
                <template #default="scope">
                  <span>{{ scope.row.analysisTime || 0 }}秒</span>
                </template>
              </el-table-column>

              <el-table-column label="创建时间" width="160">
                <template #default="scope">
                  {{ formatDate(scope.row.createdAt) }}
                </template>
              </el-table-column>

              <el-table-column label="操作" width="180" fixed="right">
                <template #default="scope">
                  <el-button-group>
                    <el-button
                      size="small"
                      title="加载此分析"
                      @click="loadFlow(scope.row)"
                    >
                      加载
                    </el-button>
                    <el-button
                      size="small"
                      type="success"
                      title="导出JSON"
                      @click="exportFlow(scope.row, 'json')"
                    >
                      JSON
                    </el-button>
                    <el-button
                      size="small"
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
      savedFlows: []
    }
  },

  mounted() {
    this.loadSavedFlows()
  },

  methods: {
    loadSavedFlows() {
      const flows = localStorage.getItem('flowviz_saved_flows')
      console.log('📦 Raw localStorage data:', flows)
      
      if (flows) {
        try {
          this.savedFlows = JSON.parse(flows)
          console.log('✅ Parsed flows:', this.savedFlows)
          console.log('📊 Flow count:', this.savedFlows.length)
          
          // 打印每条记录的详细信息
          this.savedFlows.forEach((flow, index) => {
            console.log(`\n📝 Flow ${index + 1}:`)
            console.log('  - ID:', flow.id)
            console.log('  - Title:', flow.title)
            console.log('  - InputType:', flow.inputType)
            console.log('  - InputValue:', flow.inputValue)
            console.log('  - Nodes:', flow.nodes ? flow.nodes.length : 0)
            console.log('  - Edges:', flow.edges ? flow.edges.length : 0)
            console.log('  - AnalysisTime:', flow.analysisTime)
            console.log('  - CreatedAt:', flow.createdAt)
          })
        } catch (error) {
          console.error('❌ 解析flows失败:', error)
          this.savedFlows = []
        }
      } else {
        console.log('⚠️ localStorage中没有保存的flows')
        this.savedFlows = []
      }
    },

    formatDate(dateStr) {
      if (!dateStr) return '-'
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
</style>
