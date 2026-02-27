<template>
  <div class="attck-matrix-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">MITRE ATT&CK 企业矩阵</h1>
        <p class="page-description">基于MITRE ATT&CK® 框架的恶意软件API分析平台</p>
        <div class="header-actions">
          <el-button-group>
            <el-button
              :type="viewMode === 'grid' ? 'primary' : 'default'"
              icon="el-icon-grid"
              size="small"
              @click="viewMode = 'grid'"
            >
              网格视图
            </el-button>
            <el-button
              :type="viewMode === 'table' ? 'primary' : 'default'"
              icon="el-icon-s-grid"
              size="small"
              @click="viewMode = 'table'"
            >
              表格视图
            </el-button>
          </el-button-group>
          <el-button
            type="primary"
            icon="el-icon-refresh"
            :loading="loading"
            size="small"
            @click="loadMatrixData"
          >
            刷新数据
          </el-button>
          <el-button
            type="info"
            icon="el-icon-info"
            size="small"
            @click="showDataInfo"
          >
            数据信息
          </el-button>
        </div>
      </div>
    </div>

    <!-- 统计信息 -->
    <div class="stats-grid">
      <el-card shadow="hover" class="stat-card">
        <div class="stat-content">
          <div class="stat-value">{{ stats.totalTechniques }}</div>
          <div class="stat-label">总技术数</div>
        </div>
      </el-card>
      <el-card shadow="hover" class="stat-card">
        <div class="stat-content">
          <div class="stat-value">{{ stats.techniquesWithFunctions }}</div>
          <div class="stat-label">已实现技术</div>
        </div>
      </el-card>
      <el-card shadow="hover" class="stat-card">
        <div class="stat-content">
          <div class="stat-value">{{ stats.totalFunctions }}</div>
          <div class="stat-label">总函数数</div>
        </div>
      </el-card>
      <el-card shadow="hover" class="stat-card">
        <div class="stat-content">
          <div class="stat-value">{{ stats.coverage }}%</div>
          <div class="stat-label">覆盖率</div>
        </div>
      </el-card>
    </div>

    <!-- 数据加载状态提示 -->
    <div v-if="dataSource === 'mock'" class="data-source-warning">
      <el-alert
        title="数据源提示"
        type="warning"
        :description="dataSourceMessage"
        show-icon
        :closable="false"
      />
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <div class="loading-content">
        <el-progress
          type="circle"
          :percentage="loadProgress"
          :status="loadStatus"
          :width="80"
        />
        <p>{{ loadMessage }}</p>
      </div>
    </div>

    <!-- 网格视图 -->
    <div v-else-if="viewMode === 'grid' && matrixColumns.length > 0" class="tactics-grid">
      <el-card
        v-for="tactic in matrixColumns"
        :key="tactic.id"
        class="tactic-card"
        shadow="hover"
        :body-style="{ padding: '0' }"
      >
        <!-- 战术头部 -->
        <div class="tactic-header" :style="{ backgroundColor: tactic.color }">
          <div class="tactic-id">{{ tactic.id }}</div>
          <div class="tactic-name">{{ tactic.name_cn }}</div>
          <div class="tactic-name-en">{{ tactic.name_en }}</div>
          <div class="tactic-count">
            <el-badge :value="tactic.techniques.length" type="primary" />
          </div>
        </div>

        <!-- 技术列表 -->
        <div class="tactic-body">
          <div
            v-for="technique in tactic.techniques"
            :key="technique.technique_id"
            class="technique-wrapper"
          >
            <!-- 主技术项 -->
            <div
              class="technique-item"
              @click="showTechniqueDetail(technique, tactic)"
            >
              <div class="technique-main">
                <div class="technique-id">{{ technique.technique_id }}</div>
                <div class="technique-name">{{ technique.technique_name }}</div>
              </div>
              <div class="technique-actions">
                <el-badge
                  v-if="technique.function_count > 0"
                  :value="technique.function_count"
                  :type="getBadgeType(technique.function_count)"
                  class="function-count"
                />
                <el-button
                  v-if="technique.sub_techniques && technique.sub_techniques.length > 0"
                  type="text"
                  size="mini"
                  :icon="expandedTechniques.includes(technique.technique_id) ? 'el-icon-caret-top' : 'el-icon-caret-bottom'"
                  @click.stop="toggleTechnique(technique.technique_id)"
                />
              </div>
            </div>

            <!-- 子技术列表 -->
            <div
              v-if="expandedTechniques.includes(technique.technique_id) && technique.sub_techniques.length > 0"
              class="sub-techniques-container"
            >
              <div
                v-for="subTech in technique.sub_techniques"
                :key="subTech.sub_id"
                class="sub-technique-item"
                @click="showSubTechniqueDetail(subTech, technique)"
              >
                <div class="sub-technique-info">
                  <div class="sub-technique-id">{{ subTech.sub_id }}</div>
                  <div class="sub-technique-name">{{ subTech.sub_name }}</div>
                </div>
                <el-badge
                  v-if="subTech.function_count > 0"
                  :value="subTech.function_count"
                  :type="getBadgeType(subTech.function_count)"
                  class="sub-function-count"
                />
              </div>
            </div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 表格视图 -->
    <div v-else-if="viewMode === 'table' && flattenedTechniques.length > 0" class="matrix-table-container">
      <el-card shadow="never" class="table-card">
        <div slot="header" class="table-header">
          <span>ATT&CK 技术列表 ({{ flattenedTechniques.length }} 项)</span>
          <div class="table-actions">
            <el-input
              v-model="tableSearch"
              placeholder="搜索技术ID或名称"
              prefix-icon="el-icon-search"
              size="small"
              style="width: 200px; margin-right: 10px;"
              @input="handleTableSearch"
            />
            <el-button
              size="small"
              icon="el-icon-download"
              @click="exportTableData"
            >
              导出
            </el-button>
          </div>
        </div>
        <el-table
          v-loading="tableLoading"
          :data="pagedTechniques"
          style="width: 100%"
          stripe
          :default-sort="{prop: 'tactic_id', order: 'ascending'}"
          height="calc(100vh - 400px)"
          @row-click="handleTableRowClick"
        >
          <el-table-column prop="tactic_id" label="战术ID" width="120" sortable />
          <el-table-column prop="tactic_name_cn" label="战术名称" width="150" />
          <el-table-column prop="technique_id" label="技术ID" width="120" sortable />
          <el-table-column prop="technique_name" label="技术名称" width="300" />
          <el-table-column prop="function_count" label="函数数量" width="100" sortable>
            <template #default="scope">
              <el-badge
                :value="scope.row.function_count"
                :type="getBadgeType(scope.row.function_count)"
              />
            </template>
          </el-table-column>
          <el-table-column prop="is_sub_technique" label="类型" width="80">
            <template #default="scope">
              <el-tag
                v-if="scope.row.is_sub_technique"
                type="info"
                size="small"
              >
                子技术
              </el-tag>
              <el-tag v-else type="primary" size="small">
                主技术
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="120" fixed="right">
            <template #default="scope">
              <el-button
                type="text"
                size="small"
                @click.stop="showRowDetail(scope.row)"
              >
                详情
              </el-button>
            </template>
          </el-table-column>
        </el-table>
        <div class="table-footer">
          <span>共 {{ filteredTechniques.length }} 条数据</span>
          <el-pagination
            v-if="filteredTechniques.length > pageSize"
            :current-page="currentPage"
            :page-sizes="[20, 50, 100, 200]"
            :page-size="pageSize"
            layout="total, sizes, prev, pager, next, jumper"
            :total="filteredTechniques.length"
            @size-change="handleSizeChange"
            @current-change="handleCurrentChange"
          />
        </div>
      </el-card>
    </div>

    <!-- 空状态 -->
    <div v-else-if="!loading && matrixColumns.length === 0" class="empty-state">
      <el-empty description="暂无矩阵数据">
        <el-button type="primary" @click="loadMatrixData">重新加载</el-button>
      </el-empty>
    </div>

    <!-- 技术详情弹窗 -->
    <el-dialog
      :title="`${currentTechnique.technique_id} - ${currentTechnique.technique_name}`"
      v-model="techDialogVisible"
      width="900px"
      :close-on-click-modal="false"
      @close="resetDialog"
    >
      <div v-if="currentTechnique.technique_id">
        <!-- 技术头部 -->
        <div class="technique-detail-header">
          <h2 class="tech-title">{{ currentTechnique.technique_name }}</h2>
          <div class="tech-subtitle">{{ currentTactic.name_cn }}</div>
          <div class="tech-id-badge">{{ currentTechnique.technique_id }}</div>
          <div style="margin-top: 10px;">
            <el-badge
              v-if="currentTechnique.function_count > 0"
              :value="currentTechnique.function_count + '个函数'"
              type="success"
              class="function-count-badge"
            />
          </div>
        </div>

        <!-- 技术信息 -->
        <el-tabs v-model="activeTab" class="technique-tabs" @tab-click="handleTabClick">
          <el-tab-pane label="基本信息" name="basic">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="技术ID">{{ currentTechnique.technique_id }}</el-descriptions-item>
              <el-descriptions-item label="技术名称">{{ currentTechnique.technique_name }}</el-descriptions-item>
              <el-descriptions-item label="所属战术">{{ currentTactic.name_cn }}</el-descriptions-item>
              <el-descriptions-item label="函数数量">{{ currentTechnique.function_count || 0 }}</el-descriptions-item>
              <el-descriptions-item v-if="currentTechnique.parent_technique_id" label="父技术" :span="2">
                <el-button
                  type="text"
                  @click="goToTechnique(currentTechnique.parent_technique_id)"
                >
                  {{ currentTechnique.parent_technique_id }}
                </el-button>
              </el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>

          <el-tab-pane label="相关函数" name="functions">
            <div v-if="relatedFunctions.length > 0">
              <el-table 
                v-loading="functionsLoading" 
                :data="relatedFunctions" 
                style="width: 100%"
                @row-click="handleFunctionRowClick"
              >
                <el-table-column label="程序API名称" prop="alias" width="200">
                  <template #default="scope">
                    <el-button type="text" @click="goToFunctionDetail(scope.row)">
                      {{ scope.row.alias }}
                    </el-button>
                  </template>
                </el-table-column>
                
                <el-table-column label="简短描述" prop="summary" min-width="300" show-overflow-tooltip>
                  <template #default="scope">
                    <span>{{ scope.row.summary || '暂无描述' }}</span>
                  </template>
                </el-table-column>
                
                <el-table-column label="根函数" prop="root_function" width="200">
                  <template #default="scope">
                    <el-tag type="info" size="small">{{ scope.row.root_function || '无' }}</el-tag>
                  </template>
                </el-table-column>
                
                <el-table-column label="操作" width="120">
                  <template #default="scope">
                    <el-button type="text" size="small" @click="goToFunctionDetail(scope.row)">
                      详情
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
              
              <!-- 分页 -->
              <div v-if="relatedFunctionsPagination.total > relatedFunctionsPagination.page_size" 
                   class="function-pagination">
                <el-pagination
                  small
                  layout="prev, pager, next"
                  :current-page="relatedFunctionsPagination.page"
                  :page-size="relatedFunctionsPagination.page_size"
                  :total="relatedFunctionsPagination.total"
                  @current-change="handleFunctionPageChange"
                />
              </div>
            </div>
            <div v-else class="empty-state">
              <el-empty description="暂无相关函数" />
            </div>
          </el-tab-pane>

          <el-tab-pane v-if="currentTechnique.sub_techniques && currentTechnique.sub_techniques.length > 0" 
                       label="子技术" name="subtechniques">
            <div class="subtechniques-grid">
              <el-card
                v-for="subTech in currentTechnique.sub_techniques"
                :key="subTech.sub_id"
                class="subtechnique-card"
                shadow="hover"
                @click.native="showSubTechniqueDetail(subTech, currentTechnique)"
              >
                <div class="subtechnique-header">
                  <div class="subtechnique-id">{{ subTech.sub_id }}</div>
                  <el-badge
                    v-if="subTech.function_count > 0"
                    :value="subTech.function_count"
                    :type="getBadgeType(subTech.function_count)"
                  />
                </div>
                <div class="subtechnique-name">{{ subTech.sub_name }}</div>
              </el-card>
            </div>
          </el-tab-pane>
        </el-tabs>
      </div>
    </el-dialog>

    <!-- 数据信息弹窗 -->
    <el-dialog
      title="数据源信息"
      v-model="dataInfoVisible"
      width="500px"
    >
      <el-descriptions :column="1" border>
        <el-descriptions-item label="数据源">
          <el-tag :type="dataSource === 'mock' ? 'warning' : 'success'">
            {{ dataSource === 'mock' ? '模拟数据' : '真实数据' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="战术数量">{{ matrixColumns.length }}</el-descriptions-item>
        <el-descriptions-item label="总技术数">{{ stats.totalTechniques }}</el-descriptions-item>
        <el-descriptions-item label="已实现技术">{{ stats.techniquesWithFunctions }}</el-descriptions-item>
        <el-descriptions-item label="总函数数">{{ stats.totalFunctions }}</el-descriptions-item>
        <el-descriptions-item label="覆盖率">{{ stats.coverage }}%</el-descriptions-item>
        <el-descriptions-item label="加载时间">{{ loadTime }}ms</el-descriptions-item>
      </el-descriptions>
      <div slot="footer">
        <el-button @click="dataInfoVisible = false">关闭</el-button>
        <el-button type="primary" @click="clearCache">清除缓存</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import attckApi from '@/api/attck'
import { exportJsonToExcel } from '@/utils/exportExcel'

export default {
  name: 'AttckMatrix',
  data() {
    return {
      // 加载状态
      loading: false,
      loadProgress: 0,
      loadStatus: 'success',
      loadMessage: '正在加载数据...',
      loadTime: 0,

      // 数据源信息
      dataSource: 'unknown',
      dataSourceMessage: '',
      dataInfoVisible: false,

      // 视图模式
      viewMode: 'grid',

      // 矩阵数据
      matrixData: {},
      matrixColumns: [],
      flattenedTechniques: [],
      filteredTechniques: [],

      // 表格相关
      tableSearch: '',
      tableLoading: false,
      currentPage: 1,
      pageSize: 20,

      // 统计数据
      stats: {
        totalTechniques: 0,
        techniquesWithFunctions: 0,
        totalFunctions: 0,
        coverage: 0
      },

      // 当前选中的技术
      currentTechnique: {
        technique_id: '',
        technique_name: '',
        function_count: 0,
        sub_techniques: [],
        parent_technique_id: ''
      },
      currentTactic: {
        id: '',
        name_cn: '',
        name_en: '',
        color: '#666666'
      },

      // 弹窗控制
      techDialogVisible: false,
      activeTab: 'basic',

      // 相关函数列表
      relatedFunctions: [],
      functionsLoading: false,
      relatedFunctionsPagination: {
        page: 1,
        page_size: 10,
        total: 0,
        total_pages: 0
      },

      // 展开的技术ID列表
      expandedTechniques: [],

      // 战术颜色配置
      tacticColors: {
        'TA0043': '#c5c5c5', // 侦察
        'TA0042': '#ff9800', // 资源开发
        'TA0001': '#d32f2f', // 初始访问
        'TA0002': '#f44336', // 执行
        'TA0003': '#ff5722', // 持久化
        'TA0004': '#ff6f00', // 权限提升
        'TA0005': '#9c27b0', // 防御规避
        'TA0006': '#ff8f00', // 凭证访问
        'TA0007': '#795548', // 发现
        'TA0008': '#607d8b', // 横向移动
        'TA0009': '#e91e63', // 收集
        'TA0010': '#3f51b5', // 命令与控制
        'TA0040': '#f44336', // 数据渗出
        'TA0037': '#4caf50' // 影响
      }
    }
  },
  computed: {
    pagedTechniques() {
      const start = (this.currentPage - 1) * this.pageSize
      const end = start + this.pageSize
      return this.filteredTechniques.slice(start, end)
    }
  },
  created() {
    this.loadMatrixData()
  },
  methods: {
    // 加载矩阵数据
    async loadMatrixData() {
      const startTime = Date.now()

      try {
        this.loading = true
        this.loadProgress = 0
        this.loadMessage = '正在加载静态矩阵数据...'

        // 1. 加载静态矩阵结构
        this.loadProgress = 30
        this.loadMessage = '正在解析矩阵结构...'

        const matrixData = await attckApi.getAttckMatrix()
        this.matrixData = matrixData
        
        // 判断数据源
        if (matrixData.isMock) {
          this.dataSource = 'mock'
          this.dataSourceMessage = '当前使用模拟数据，请检查静态文件是否正确放置。'
        } else {
          this.dataSource = 'real'
          this.dataSourceMessage = '使用真实矩阵数据。'
        }

        this.loadProgress = 50
        this.loadMessage = '正在处理数据格式...'

        // 2. 尝试加载统计数据（如果有后端API）
        let functionStats = {}
        try {
          this.loadMessage = '正在获取函数统计数据...'
          const statsRes = await attckApi.getMatrixStats()
          functionStats = statsRes.function_stats || statsRes || {}
          // 检查是否有数据
          if (!functionStats || Object.keys(functionStats).length === 0) {
            console.warn('没有获取到矩阵统计数据')
          }
          this.loadProgress = 70
        } catch (error) {
          console.warn('获取矩阵统计数据失败，使用默认值:', error)
          this.loadProgress = 70
        }

        // 3. 转换数据格式
        this.loadMessage = '正在转换数据格式...'
        this.transformMatrixData(this.matrixData, functionStats)
        this.loadProgress = 90

        // 4. 计算统计数据
        this.calculateStatistics()

        this.loadProgress = 100
        this.loadMessage = '加载完成！'
        this.loadStatus = 'success'

        this.loadTime = Date.now() - startTime
        console.log(`✅ 矩阵数据加载完成，耗时 ${this.loadTime}ms`)
      } catch (error) {
        console.error('加载矩阵数据失败:', error)
        this.$message.error('加载ATT&CK矩阵数据失败')
        this.loadStatus = 'exception'
        this.loadMessage = '加载失败，请检查网络或文件配置'

        // 使用模拟数据
        this.dataSource = 'mock'
        this.dataSourceMessage = '数据加载失败，使用模拟数据。'
        this.matrixColumns = []
        this.flattenedTechniques = []
        this.filteredTechniques = []
      } finally {
        this.loading = false
        setTimeout(() => {
          this.loadProgress = 0
          this.loadMessage = '正在加载数据...'
        }, 1000)
      }
    },

    // 转换矩阵数据格式
    transformMatrixData(matrixData, functionStats) {
      this.matrixColumns = []
      this.flattenedTechniques = []

      Object.entries(matrixData).forEach(([tacticId, tacticData]) => {
        // 跳过模拟数据的标记属性
        if (tacticId === 'isMock') return

        const column = {
          id: tacticId,
          name_en: tacticData.tactic_name_en || '',
          name_cn: tacticData.tactic_name_cn || '',
          color: this.tacticColors[tacticId] || '#666666',
          techniques: []
        }

        tacticData.techniques.forEach(techObj => {
          let techniqueId = ''
          let techniqueName = ''
          let subTechniques = []

          Object.entries(techObj).forEach(([key, value]) => {
            if (key === 'sub') {
              // 子技术处理
              subTechniques = value.map(sub => ({
                sub_id: Object.keys(sub)[0],    // 子技术ID
                sub_name: sub[Object.keys(sub)[0]], // 子技术名称
                function_count: functionStats[Object.keys(sub)[0]] || 0,
                parent_technique_id: techniqueId
              }))
            } else {
              techniqueId = key
              techniqueName = value
            }
          })

          const functionCount = functionStats[techniqueId] || 0

          column.techniques.push({
            technique_id: techniqueId,
            technique_name: techniqueName,
            function_count: functionCount,
            sub_techniques: subTechniques,
            is_main_technique: true
          })

          this.flattenedTechniques.push({
            tactic_id: tacticId,
            tactic_name_cn: column.name_cn,
            technique_id: techniqueId,
            technique_name: techniqueName,
            function_count: functionCount,
            is_sub_technique: false,
            is_main_technique: true
          })

          subTechniques.forEach(sub => {
            this.flattenedTechniques.push({
              tactic_id: tacticId,
              tactic_name_cn: column.name_cn,
              technique_id: sub.sub_id,
              technique_name: sub.sub_name,
              function_count: sub.function_count,
              is_sub_technique: true,
              is_main_technique: false,
              parent_technique_id: techniqueId
            })
          })
        })

        this.matrixColumns.push(column)
      })

      // 按战术ID排序
      this.matrixColumns.sort((a, b) => a.id.localeCompare(b.id))

      // 初始化过滤后的数据
      this.filteredTechniques = [...this.flattenedTechniques]
    },

    // 计算统计数据
    calculateStatistics() {
      let totalTechniques = 0
      let techniquesWithFunctions = 0
      let totalFunctions = 0

      this.flattenedTechniques.forEach(tech => {
        totalTechniques++
        totalFunctions += tech.function_count || 0
        if (tech.function_count > 0) {
          techniquesWithFunctions++
        }
      })

      const coverage = totalTechniques > 0 ? Math.round((techniquesWithFunctions / totalTechniques) * 100) : 0

      this.stats = {
        totalTechniques,
        techniquesWithFunctions,
        totalFunctions,
        coverage
      }
    },

    // 表格搜索
    handleTableSearch() {
      if (!this.tableSearch.trim()) {
        this.filteredTechniques = [...this.flattenedTechniques]
        return
      }

      const searchTerm = this.tableSearch.toLowerCase()
      this.filteredTechniques = this.flattenedTechniques.filter(tech => {
        return tech.technique_id.toLowerCase().includes(searchTerm) ||
               tech.technique_name.toLowerCase().includes(searchTerm) ||
               tech.tactic_name_cn.toLowerCase().includes(searchTerm)
      })

      this.currentPage = 1
    },

    // 分页大小改变
    handleSizeChange(val) {
      this.pageSize = val
      this.currentPage = 1
    },

    // 当前页改变
    handleCurrentChange(val) {
      this.currentPage = val
    },

    // 导出表格数据
    exportTableData() {
      const exportData = this.filteredTechniques.map(tech => ({
        '战术ID': tech.tactic_id,
        '战术名称': tech.tactic_name_cn,
        '技术ID': tech.technique_id,
        '技术名称': tech.technique_name,
        '函数数量': tech.function_count,
        '类型': tech.is_sub_technique ? '子技术' : '主技术',
        '父技术ID': tech.parent_technique_id || ''
      }))

      const fileName = `attck_matrix_${new Date().toISOString().slice(0, 10)}.xlsx`
      exportJsonToExcel(exportData, fileName)
      this.$message.success('导出成功')
    },

    // 切换技术展开状态
    toggleTechnique(techniqueId) {
      const index = this.expandedTechniques.indexOf(techniqueId)
      if (index > -1) {
        this.expandedTechniques.splice(index, 1)
      } else {
        this.expandedTechniques.push(techniqueId)
      }
    },

    // 显示技术详情
    async showTechniqueDetail(technique, tactic = null) {
      // 跳转到技术详情页面
      this.$router.push({
        path: `/attck/technique/${technique.technique_id}`,
        query: {
          tactic_id: tactic?.id,
          tactic_name: tactic?.name_cn
        }
      })
    },

    // 处理标签页点击
    async handleTabClick(tab) {
      if (tab.name === 'functions' && this.currentTechnique.function_count > 0 && this.relatedFunctions.length === 0) {
        await this.loadRelatedFunctions()
      }
    },

    // 加载相关函数
    async loadRelatedFunctions() {
      this.functionsLoading = true
      try {
        const res = await attckApi.getTechniqueFunctionsDetail(this.currentTechnique.technique_id, {
          page: this.relatedFunctionsPagination.page,
          page_size: this.relatedFunctionsPagination.page_size
        })
        
        if (res.success) {
          this.relatedFunctions = res.data || []
          this.relatedFunctionsPagination = {
            page: res.pagination?.page || 1,
            page_size: res.pagination?.page_size || 10,
            total: res.pagination?.total || this.relatedFunctions.length,
            total_pages: res.pagination?.total_pages || 1
          }
        } else {
          console.error('加载相关函数失败:', res.error)
          this.relatedFunctions = []
          this.$message.warning('加载相关函数失败')
        }
      } catch (error) {
        console.error('加载相关函数失败:', error)
        this.relatedFunctions = []
        this.$message.warning('加载相关函数失败')
      } finally {
        this.functionsLoading = false
      }
    },

    // 函数分页改变
    handleFunctionPageChange(page) {
      this.relatedFunctionsPagination.page = page
      this.loadRelatedFunctions()
    },

    // 函数行点击
    handleFunctionRowClick(row) {
      // 这里可以处理函数行的点击事件
      console.log('点击函数行:', row)
    },

    // 跳转到函数详情
    goToFunctionDetail(row) {
      // 阻止事件冒泡
      if (event) {
        event.stopPropagation()
      }
      
      this.$router.push({
        path: '/attck/function-detail',
        query: {
          id: row.id,
          file_name: row.file_name,
          alias: row.alias
        }
      })
    },

    // 显示子技术详情
    showSubTechniqueDetail(subTech, parentTechnique) {
      // 跳转到技术详情页面
      this.$router.push({
        path: `/attck/technique/${subTech.sub_id}`,
        query: {
          parent_technique_id: parentTechnique.technique_id,
          parent_technique_name: parentTechnique.technique_name
        }
      })
    },

    // 处理表格行点击
    handleTableRowClick(row) {
      this.showRowDetail(row)
    },

    // 显示行详情
    showRowDetail(row) {
      if (row.is_sub_technique) {
        // 跳转到子技术详情页面
        this.$router.push({
          path: `/attck/technique/${row.technique_id}`,
          query: {
            parent_technique_id: row.parent_technique_id,
            tactic_id: row.tactic_id,
            tactic_name: row.tactic_name_cn
          }
        })
      } else {
        // 跳转到主技术详情页面
        this.$router.push({
          path: `/attck/technique/${row.technique_id}`,
          query: {
            tactic_id: row.tactic_id,
            tactic_name: row.tactic_name_cn
          }
        })
      }
    },

    // 查找技术
    findTechniqueById(techniqueId) {
      for (const tactic of this.matrixColumns) {
        const technique = tactic.techniques.find(t => t.technique_id === techniqueId)
        if (technique) return technique
      }
      return null
    },

    // 查找父技术
    findParentTechnique(subTechniqueId) {
      for (const tactic of this.matrixColumns) {
        for (const tech of tactic.techniques) {
          const subTech = tech.sub_techniques.find(sub => sub.sub_id === subTechniqueId)
          if (subTech) {
            return tech
          }
        }
      }
      return null
    },

    // 跳转到技术详情页
    goToTechnique(techniqueId) {
      this.$router.push(`/attck/technique/${techniqueId}`)
    },

    // 显示数据信息
    showDataInfo() {
      this.dataInfoVisible = true
    },

    // 清除缓存
    clearCache() {
      attckApi.clearMatrixCache()
      this.$message.success('已清除缓存，下次加载将重新获取数据')
      this.dataInfoVisible = false
    },

    // 重置弹窗
    resetDialog() {
      this.currentTechnique = {
        technique_id: '',
        technique_name: '',
        function_count: 0,
        sub_techniques: [],
        parent_technique_id: ''
      }
      this.currentTactic = {
        id: '',
        name_cn: '',
        name_en: '',
        color: '#666666'
      }
      this.relatedFunctions = []
      this.activeTab = 'basic'
      this.functionsLoading = false
      this.relatedFunctionsPagination = {
        page: 1,
        page_size: 10,
        total: 0,
        total_pages: 0
      }
    },

    // 获取徽章类型
    getBadgeType(count) {
      if (count === 0) return 'info'
      if (count <= 5) return 'success'
      if (count <= 10) return 'warning'
      return 'danger'
    }
  }
}
</script>

<style scoped>
.attck-matrix-page {
  padding: 20px;
  min-height: calc(100vh - 84px);
  background-color: #f5f7fa;
}

/* 页面头部 */
.page-header {
  margin-bottom: 20px;
}

.header-content {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.page-title {
  font-size: 24px;
  font-weight: 700;
  color: #303133;
  margin: 0 0 8px 0;
}

.page-description {
  color: #606266;
  font-size: 14px;
  margin: 0 0 16px 0;
  line-height: 1.5;
}

.header-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

/* 数据源警告 */
.data-source-warning {
  margin-bottom: 20px;
}

/* 统计网格 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}

.stat-card {
  border-radius: 8px;
  border: none;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.stat-content {
  text-align: center;
  padding: 20px;
}

.stat-value {
  font-size: 32px;
  font-weight: 700;
  color: #409eff;
  margin-bottom: 8px;
  line-height: 1;
}

.stat-label {
  font-size: 14px;
  color: #909399;
}

/* 加载状态 */
.loading-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.loading-content {
  text-align: center;
}

.loading-content p {
  margin-top: 16px;
  color: #606266;
}

/* 空状态 */
.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

/* 战术网格 */
.tactics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.tactic-card {
  border-radius: 8px;
  overflow: hidden;
  transition: all 0.3s ease;
}

.tactic-card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

.tactic-header {
  padding: 16px;
  color: white;
  position: relative;
}

.tactic-id {
  font-size: 12px;
  opacity: 0.9;
  font-weight: 500;
}

.tactic-name {
  font-size: 16px;
  font-weight: 600;
  margin: 4px 0;
  line-height: 1.2;
}

.tactic-name-en {
  font-size: 12px;
  opacity: 0.8;
  font-style: italic;
}

.tactic-count {
  position: absolute;
  top: 16px;
  right: 16px;
}

.tactic-body {
  padding: 16px;
  max-height: 400px;
  overflow-y: auto;
}

/* 技术包装器 */
.technique-wrapper {
  margin-bottom: 8px;
}

/* 技术项 */
.technique-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  border-radius: 4px;
  background: #fafafa;
  cursor: pointer;
  transition: all 0.2s ease;
  border-left: 3px solid transparent;
}

.technique-item:hover {
  background: #ecf5ff;
  border-left-color: #409eff;
  transform: translateX(4px);
}

.technique-main {
  flex: 1;
  min-width: 0;
  padding-right: 8px;
}

.technique-id {
  font-size: 12px;
  color: #409eff;
  font-weight: 600;
  margin-bottom: 4px;
}

.technique-name {
  font-size: 13px;
  color: #303133;
  line-height: 1.3;
  word-break: break-word;
}

.technique-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.function-count {
  transform: scale(0.8);
}

/* 子技术容器 */
.sub-techniques-container {
  margin-left: 16px;
  padding-left: 8px;
  border-left: 2px solid #e4e7ed;
}

.sub-technique-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 10px;
  border-radius: 4px;
  background: #f5f7fa;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 4px;
  border-left: 2px solid transparent;
}

.sub-technique-item:hover {
  background: #f0f9ff;
  border-left-color: #67c23a;
}

.sub-technique-info {
  flex: 1;
  min-width: 0;
}

.sub-technique-id {
  font-size: 11px;
  color: #909399;
  font-weight: 600;
  margin-bottom: 2px;
}

.sub-technique-name {
  font-size: 12px;
  color: #606266;
  line-height: 1.2;
}

.sub-function-count {
  transform: scale(0.7);
}

/* 表格容器 */
.matrix-table-container {
  margin-top: 20px;
}

.table-card {
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 16px;
  color: #303133;
}

.table-actions {
  display: flex;
  align-items: center;
}

.table-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0;
  margin-top: 16px;
  border-top: 1px solid #ebeef5;
  color: #909399;
  font-size: 14px;
}

/* 技术详情弹窗 */
.technique-detail-header {
  background: linear-gradient(135deg, #409eff 0%, #337ecc 100%);
  color: white;
  padding: 24px;
  margin: -20px -20px 20px -20px;
  border-radius: 8px 8px 0 0;
}

.tech-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 8px;
  line-height: 1.2;
}

.tech-subtitle {
  font-size: 16px;
  opacity: 0.9;
  margin-bottom: 8px;
}

.tech-id-badge {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 14px;
  display: inline-block;
  font-weight: 500;
}

.function-count-badge {
  margin-top: 8px;
}

/* 函数分页 */
.function-pagination {
  margin-top: 16px;
  display: flex;
  justify-content: center;
}

/* 子技术网格 */
.subtechniques-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.subtechnique-card {
  cursor: pointer;
  transition: all 0.3s ease;
  border-radius: 6px;
  border: 1px solid #ebeef5;
}

.subtechnique-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  border-color: #409eff;
}

.subtechnique-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.subtechnique-id {
  font-size: 12px;
  color: #409eff;
  font-weight: 600;
}

.subtechnique-name {
  font-size: 13px;
  color: #303133;
  line-height: 1.4;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .tactics-grid {
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  }
}

@media (max-width: 992px) {
  .tactics-grid {
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  }

  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .attck-matrix-page {
    padding: 12px;
  }

  .tactics-grid {
    grid-template-columns: 1fr;
  }

  .header-actions {
    flex-direction: column;
    align-items: stretch;
  }

  .table-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .table-actions {
    width: 100%;
    justify-content: space-between;
  }
}

@media (max-width: 576px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }

  .page-title {
    font-size: 20px;
  }

  .stat-value {
    font-size: 28px;
  }
}
</style>