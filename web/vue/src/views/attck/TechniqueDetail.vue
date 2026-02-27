<template>
  <div class="technique-detail-page">
    <!-- 返回按钮 -->
    <div class="back-button-container">
      <el-button type="default" icon="el-icon-arrow-left" @click="goBack">
        返回上一页
      </el-button>
    </div>

    <div v-if="loading" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <div v-else-if="error" class="error-container">
      <el-alert
        title="加载失败"
        :description="error"
        type="error"
        show-icon
      />
      <el-button type="primary" style="margin-top: 20px;" @click="loadTechniqueDetail">
        重新加载
      </el-button>
    </div>

    <div v-else-if="technique" class="technique-content">
      <!-- 技术头部 -->
      <div class="technique-header">
        <el-tag type="primary" size="large">{{ technique.technique_id }}</el-tag>
        <h1 class="technique-title">{{ technique.technique_name }}</h1>
      </div>

      <!-- 基本信息 -->
      <el-card class="basic-info-card">
        <el-row :gutter="20">
          <el-col :span="6">
            <el-descriptions :column="1">
              <el-descriptions-item label="技术编号">{{ technique.technique_id }}</el-descriptions-item>
              <el-descriptions-item label="战术分类">
                <el-tag type="info">{{ technique.tactic_name || '未知' }}</el-tag>
              </el-descriptions-item>

            </el-descriptions>
          </el-col>
          <el-col :span="6">
            <el-descriptions :column="1">
              <el-descriptions-item label="技术名称">{{ technique.technique_name }}</el-descriptions-item>
              <el-descriptions-item label="关联函数数量">
                <el-badge :value="technique.function_count || 0" type="success" />
              </el-descriptions-item>
              <el-descriptions-item v-if="technique.related_technique_id" label="关联技术">
                {{ technique.related_technique_id }}
              </el-descriptions-item>
            </el-descriptions>
          </el-col>
        </el-row>
      </el-card>

      <!-- 技术描述 -->
      <el-card class="description-card">
        <div slot="header" class="card-header">
          <span>技术描述</span>
        </div>
        <div v-if="technique.chinese_description" class="chinese-description">
          {{ technique.chinese_description }}
        </div>
        <div v-else-if="technique.description" class="description-content">
          {{ technique.description }}
        </div>
        <div v-else class="empty-description">
          暂无详细描述
        </div>
        <div v-if="technique.mitre_url" class="mitre-link">
          <el-divider>MITRE ATT&CK官方链接</el-divider>
          <a :href="technique.mitre_url" target="_blank" rel="noopener noreferrer">
            {{ technique.mitre_url }}
          </a>
        </div>
      </el-card>

      <!-- 相关函数 -->
      <el-card v-if="technique.function_count > 0" class="functions-card">
        <div slot="header" class="card-header">
          <span>Procedure Examples ({{ functions.length }}个函数)</span>
        </div>
        <el-table :data="functions" style="width: 100%">
          <el-table-column prop="alias" label="程序API名称" width="250">
            <template #default="scope">
              <el-button type="text" @click="goToFunctionDetail(scope.row)">
                {{ scope.row.alias }}
              </el-button>
            </template>
          </el-table-column>
          <el-table-column prop="summary" label="简短描述" min-width="300" show-overflow-tooltip />
          <el-table-column prop="root_function" label="根函数" width="150">
            <template #default="scope">
              <el-tag type="info" size="small">{{ scope.row.root_function || '无' }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="status" label="状态" width="80">
            <template #default="scope">
              <el-tag :type="scope.row.status === 'ok' ? 'success' : 'danger'" size="small">
                {{ scope.row.status }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="100">
            <template #default="scope">
              <el-button type="text" @click="goToFunctionDetail(scope.row)">
                详情
              </el-button>
            </template>
          </el-table-column>
        </el-table>
        
        <!-- 分页 -->
        <div v-if="functions.length > 0" class="pagination-container">
          <el-pagination
            background
            layout="prev, pager, next"
            :current-page="currentPage"
            :page-size="pageSize"
            :total="totalFunctions"
            @current-change="handlePageChange"
          />
        </div>
      </el-card>

      <!-- 子技术 -->
      <el-card v-if="subTechniques.length > 0" class="subtechniques-card">
        <div slot="header" class="card-header">
          <span>子技术 ({{ subTechniques.length }})</span>
        </div>
        <div class="subtechniques-list">
          <div
            v-for="sub in subTechniques"
            :key="sub.sub_id"
            class="subtechnique-item"
            @click="goToSubTechnique(sub.sub_id)"
          >
            <div class="subtechnique-info">
              <div class="subtechnique-id">{{ sub.sub_id }}</div>
              <div class="subtechnique-name">{{ sub.sub_name }}</div>
            </div>
            <el-badge v-if="sub.function_count > 0" :value="sub.function_count" type="info" />
          </div>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script>
import attckApi from '@/api/attck'

export default {
  name: 'TechniqueDetail',
  data() {
    return {
      loading: false,
      error: null,
      technique: null,
      functions: [],
      subTechniques: [],
      currentPage: 1,
      pageSize: 20,
      totalFunctions: 0
    }
  },
  computed: {
    techniqueId() {
      return this.$route.params.id
    }
  },
  watch: {
    '$route.params.id': function(newId) {
      if (newId) {
        this.loadTechniqueDetail()
      }
    }
  },
  created() {
    this.loadTechniqueDetail()
  },
  methods: {
    // 加载技术详情
    async loadTechniqueDetail() {
      try {
        this.loading = true
        this.error = null

        // 1. 获取技术详情
        const techRes = await attckApi.getTechniqueDetail(this.techniqueId)
        this.technique = techRes.data || techRes

        // 2. 获取相关函数
        if (this.technique.function_count > 0) {
          await this.loadFunctions(1)
        }

        // 3. 如果当前技术有子技术，获取子技术信息
        if (this.technique.subtechniques && this.technique.subtechniques.length > 0) {
          this.subTechniques = this.technique.subtechniques.map(sub => ({
            sub_id: sub.technique_id,
            sub_name: sub.technique_name,
            function_count: sub.function_count || 0
          }))
        }
      } catch (error) {
        console.error('加载技术详情失败:', error)
        this.error = error.message || '加载技术详情失败'
      } finally {
        this.loading = false
      }
    },

    // 加载函数列表
    async loadFunctions(page) {
      try {
        const funcRes = await attckApi.getTechniqueFunctionsDetail(this.techniqueId, {
          page: page,
          page_size: this.pageSize
        })
        this.functions = funcRes.data?.data || funcRes?.data || []
        this.totalFunctions = funcRes.data?.pagination?.total || funcRes?.pagination?.total || 0
        this.currentPage = page
      } catch (error) {
        console.error('加载函数列表失败:', error)
        this.$message.error('加载函数列表失败')
      }
    },

    // 处理分页变化
    handlePageChange(page) {
      this.loadFunctions(page)
    },

    // 返回上一页
    goBack() {
      this.$router.go(-1)
    },

    // 跳转到函数详情
    goToFunctionDetail(functionObj) {
      this.$router.push({
        path: '/attck/function-detail',
        query: {
          id: functionObj.id,
          file_name: functionObj.file_name,
          alias: functionObj.alias
        }
      })
    },

    // 跳转到子技术
    goToSubTechnique(techniqueId) {
      this.$router.push(`/attck/technique/${techniqueId}`)
    },

    // AI分析代码
    async analyzeCode(functionId) {
      try {
        const res = await attckApi.analyzeCode({
          function_ids: [functionId],
          analysis_type: 'code_explanation'
        })
        console.log('AI分析结果:', res)
        this.$message.success('AI分析完成')
      } catch (error) {
        console.error('AI分析失败:', error)
        this.$message.error('AI分析失败')
      }
    }
  }
}
</script>

<style scoped>
.technique-detail-page {
  padding: 20px;
  background: #f0f2f5;
  min-height: calc(100vh - 60px);
}

/* 返回按钮 */
.back-button-container {
  margin-bottom: 20px;
}

.loading-container,
.error-container {
  padding: 40px;
  text-align: center;
}

/* 技术头部 */
.technique-header {
  margin-bottom: 20px;
}

.technique-header .el-tag {
  margin-bottom: 10px;
}

.technique-title {
  margin: 0;
  color: #262626;
  font-size: 32px;
  font-weight: 700;
  line-height: 1.2;
}

/* 基本信息卡片 */
.basic-info-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.platform-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 4px;
}

/* 技术描述卡片 */
.description-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.chinese-description {
  font-size: 14px;
  line-height: 1.8;
  color: #333;
  padding: 10px 0;
}

.description-content {
  font-size: 14px;
  line-height: 1.6;
  color: #595959;
  padding: 10px 0;
}

.empty-description {
  color: #8c8c8c;
  font-style: italic;
  padding: 20px 0;
  text-align: center;
}

.mitre-link {
  margin-top: 20px;
}

.mitre-link a {
  color: #1890ff;
  text-decoration: none;
}

.mitre-link a:hover {
  text-decoration: underline;
}

/* 函数卡片 */
.functions-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

/* 子技术卡片 */
.subtechniques-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.subtechniques-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.subtechnique-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  border-radius: 4px;
  background: #fafafa;
  cursor: pointer;
  transition: all 0.2s ease;
}

.subtechnique-item:hover {
  background: #e6f7ff;
}

.subtechnique-info {
  flex: 1;
}

.subtechnique-id {
  font-size: 12px;
  color: #666;
  font-weight: 600;
  margin-bottom: 4px;
}

.subtechnique-name {
  font-size: 13px;
  color: #333;
  line-height: 1.3;
}

.card-header {
  font-size: 16px;
  font-weight: 600;
  color: #262626;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .technique-detail-page {
    padding: 10px;
  }

  .technique-title {
    font-size: 24px;
  }

  .basic-info-card .el-row {
    flex-direction: column;
  }

  .basic-info-card .el-col {
    width: 100%;
    margin-bottom: 20px;
  }

  .pagination-container {
    justify-content: center;
  }
}
</style>
