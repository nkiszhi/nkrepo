<!-- vue/src/views/flowviz/FlowAnalysis.vue -->
<template>
  <div class="flow-analysis">
    <el-container style="height: calc(100vh - 84px);">
      <!-- 左侧控制面板 -->
      <el-aside width="320px" class="control-aside">
        <FlowControlPanel
          ref="controlPanel"
          :is-loading="isLoading"
          :is-streaming="isStreaming"
          :provider="provider"
          :model="model"
          :progress="progress"
          :has-data="hasData"
          @start-analysis="handleStartAnalysis"
          @clear-analysis="handleClearAnalysis"
          @provider-change="handleProviderChange"
          @mode-change="handleModeChange"
          @strict-mode-change="handleStrictModeChange"
          @streaming-render-change="handleStreamingRenderChange"
          @animation-change="handleAnimationChange"
          @export="handleExport"
          @test-connection="handleTestConnection"
        />
      </el-aside>

      <!-- 主流程图区域 -->
      <el-main class="flow-main">
        <FlowVisualization
          ref="flowViz"
          :nodes="visualizationNodes"
          :edges="visualizationEdges"
          :is-streaming="isStreaming"
          :streaming-render="streamingRender"
          :enable-animations="enableAnimations"
          @node-click="handleNodeClick"
          @edge-click="handleEdgeClick"
          @nodes-updated="handleNodesUpdated"
          @edges-updated="handleEdgesUpdated"
        />

        <!-- 空状态提示 -->
        <div v-if="!hasData && !isLoading && !isStreaming" class="empty-state">
          <div class="empty-state-content">
            <div class="empty-icon">
              <i class="el-icon-data-analysis" />
            </div>
            <h3>网络安全攻击流程图</h3>
            <p class="description">
              选择输入模式，输入URL或文本开始分析<br>
              支持流式可视化展示和多种导出格式
            </p>
            <div class="mode-tips">
              <div class="mode-tip url-tip">
                <i class="el-icon-link" />
                <h4>URL分析</h4>
                <p>输入技术文章、威胁报告URL，自动获取内容进行分析</p>
              </div>
              <div class="mode-tip text-tip">
                <i class="el-icon-document" />
                <h4>文本分析</h4>
                <p>直接输入技术文章文本或JSON格式数据</p>
              </div>
            </div>
          </div>
        </div>

        <!-- 流式处理进度指示器 -->
        <div v-if="isStreaming" class="streaming-indicator">
          <div class="spinner">
            <div class="bounce1" />
            <div class="bounce2" />
            <div class="bounce3" />
          </div>
          <span>正在分析中...</span>
          <span v-if="progress.message"> {{ progress.message }}</span>
          <span class="stats">
            {{ visualizationNodes.length }} 节点 / {{ visualizationEdges.length }} 边
          </span>
        </div>

        <!-- 统计信息 -->
        <div v-if="hasData && !isStreaming" class="stats-info">
          <el-tag type="info">{{ visualizationNodes.length }} 节点</el-tag>
          <el-tag type="info">{{ visualizationEdges.length }} 边</el-tag>
          <el-tag type="success">{{ analysisTime }}秒</el-tag>
          <el-tag v-if="inputType === 'url'" type="warning">URL分析</el-tag>
          <el-tag v-else type="primary">文本分析</el-tag>
        </div>

        <!-- 节点详情抽屉 -->
        <el-drawer
          title="节点详情"
          v-model="showNodeDetails"
          direction="rtl"
          size="30%"
          :before-close="handleNodeDetailsClose"
        >
          <NodeDetailPanel
            v-if="selectedNode"
            :node="selectedNode"
          />
          <div v-else class="no-data">
            <el-empty description="未选择节点" />
          </div>
        </el-drawer>

        <!-- 边详情抽屉 -->
        <el-drawer
          title="连接详情"
          v-model="showEdgeDetails"
          direction="rtl"
          size="25%"
          :before-close="handleEdgeDetailsClose"
        >
          <EdgeDetailPanel
            v-if="selectedEdge"
            :edge="selectedEdge"
            :nodes="visualizationNodes"
          />
          <div v-else class="no-data">
            <el-empty description="未选择连接" />
          </div>
        </el-drawer>
      </el-main>
    </el-container>

    <!-- 错误提示 -->
    <el-dialog
      v-if="errorMessage"
      :title="'分析错误'"
      v-model="showErrorDialog"
      width="500px"
      center
    >
      <div class="error-content">
        <i class="el-icon-error" style="color: #F56C6C; font-size: 48px; margin-bottom: 20px;" />
        <p style="color: #F56C6C; font-weight: bold; margin-bottom: 10px;">{{ errorMessage }}</p>
        <p v-if="errorDetails" style="color: #606266; font-size: 14px;">{{ errorDetails }}</p>
        <div v-if="suggestions.length > 0" class="suggestions">
          <h4>建议：</h4>
          <ul>
            <li v-for="(suggestion, index) in suggestions" :key="index">{{ suggestion }}</li>
          </ul>
        </div>
      </div>
      <span slot="footer" class="dialog-footer">
        <el-button @click="showErrorDialog = false">关闭</el-button>
        <el-button type="primary" @click="retryAnalysis">重试</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script>
import FlowControlPanel from './components/FlowControlPanel.vue'
import FlowVisualization from './components/FlowVisualization.vue'
import NodeDetailPanel from './components/NodeDetailPanel.vue'
import EdgeDetailPanel from './components/EdgeDetailPanel.vue'
import { flowvizApi } from '@/api/flowviz'

export default {
  name: 'FlowAnalysis',
  components: {
    FlowControlPanel,
    FlowVisualization,
    NodeDetailPanel,
    EdgeDetailPanel
  },
  data() {
    return {
      isLoading: false,
      isStreaming: false,
      inputType: 'text', // 'text' 或 'url'
      inputValue: '',
      provider: 'openai',
      model: 'gpt-4o',
      // 存储后端返回的原始数据
      rawNodes: [],
      rawEdges: [],
      // 传递给可视化的数据（包含位置信息）
      visualizationNodes: [],
      visualizationEdges: [],
      selectedNode: null,
      selectedEdge: null,
      showNodeDetails: false,
      showEdgeDetails: false,
      showErrorDialog: false,
      errorMessage: '',
      errorDetails: '',
      suggestions: [],
      analysisTime: 0,
      analysisStartTime: null,
      strictMode: true,
      streamingRender: true,
      enableAnimations: true,
      progress: {
        stage: '',
        message: '',
        percentage: 0,
        status: undefined  // 修复：初始化为undefined而不是空字符串
      }
    }
  },

  computed: {
    hasData() {
      return this.visualizationNodes.length > 0 || this.visualizationEdges.length > 0
    }
  },

  mounted() {
    // 设置全局回调，用于接收流式数据
    window.handleStreamData = this.handleStreamData.bind(this)

    // 加载提供商配置
    this.loadProviders()

    // 恢复设置
    this.restoreSettings()
  },

  beforeDestroy() {
    // 清理全局回调
    window.handleStreamData = null

    // 保存设置
    this.saveSettings()
  },

  methods: {
    /**
     * 加载提供商配置
     */
    async loadProviders() {
      try {
        const providers = await flowvizApi.getProviders()
        console.log('提供商配置:', providers)
      } catch (error) {
        console.warn('加载提供商配置失败:', error)
      }
    },

    /**
     * 恢复设置
     */
    restoreSettings() {
      const settings = localStorage.getItem('flowviz_settings')
      if (settings) {
        try {
          const parsed = JSON.parse(settings)
          this.strictMode = parsed.strictMode !== undefined ? parsed.strictMode : true
          this.streamingRender = parsed.streamingRender !== undefined ? parsed.streamingRender : true
          this.enableAnimations = parsed.enableAnimations !== undefined ? parsed.enableAnimations : true
          this.provider = parsed.provider || 'openai'
          this.model = parsed.model || 'gpt-4o'
        } catch (e) {
          console.error('恢复设置失败:', e)
        }
      }
    },

    /**
     * 保存设置
     */
    saveSettings() {
      const settings = {
        strictMode: this.strictMode,
        streamingRender: this.streamingRender,
        enableAnimations: this.enableAnimations,
        provider: this.provider,
        model: this.model
      }
      localStorage.setItem('flowviz_settings', JSON.stringify(settings))
    },

    /**
     * 处理提供商变更
     */
    handleProviderChange({ provider, model }) {
      console.log('🔄 提供商变更:', provider, model)
      this.provider = provider
      this.model = model
      this.saveSettings()
    },

    /**
     * 处理输入模式变更
     */
    handleModeChange(mode) {
      this.inputType = mode
    },

    /**
     * 处理严格模式变更
     */
    handleStrictModeChange(value) {
      this.strictMode = value
      this.saveSettings()
    },

    /**
     * 处理流式渲染变更
     */
    handleStreamingRenderChange(value) {
      this.streamingRender = value
      this.saveSettings()
    },

    /**
     * 处理动画变更
     */
    handleAnimationChange(value) {
      this.enableAnimations = value
      this.saveSettings()
    },

    /**
     * 开始分析
     */
    async handleStartAnalysis(data) {
      console.log('🚀 开始分析:', data)

      if (!data || !data.value) {
        this.$message.warning('请输入分析内容')
        return
      }

      // 重置状态
      this.resetAnalysisState()

      this.inputType = data.type
      this.inputValue = data.value
      this.isLoading = true
      this.isStreaming = true
      this.analysisStartTime = Date.now()

      // 显示进度 - 修复：设置有效的status值
      this.progress = {
        stage: 'initializing',
        message: '正在初始化分析引擎...',
        percentage: 5,
        status: undefined  // 分析中不使用status
      }

      // 长文本提示
      if (data.type === 'text' && data.value.length > 5000) {
        this.$message.info('检测到长文本，分析可能需要1-2分钟，请耐心等待...')
      }

      try {
        // 调用流式分析 API
        await flowvizApi.analyzeStream({
          input: data.value,
          inputType: data.type,
          provider: this.provider,
          model: this.model,
          strictMode: this.strictMode
        })

        console.log('📤 分析请求已发送，等待流式响应...')
      } catch (error) {
        console.error('❌ 分析失败:', error)
        this.handleAnalysisError(error)
      }
    },

    /**
     * 重置分析状态 - 修复第374行的问题
     */
    resetAnalysisState() {
      this.rawNodes = []
      this.rawEdges = []
      this.visualizationNodes = []
      this.visualizationEdges = []
      this.selectedNode = null
      this.selectedEdge = null
      this.showNodeDetails = false
      this.showEdgeDetails = false
      this.errorMessage = ''
      this.errorDetails = ''
      this.suggestions = []
      this.analysisTime = 0
      
      // 修复：将status设置为undefined而不是空字符串
      this.progress = {
        stage: '',
        message: '',
        percentage: 0,
        status: undefined  // 重要修复：Element UI不接受空字符串
      }
    },

    /**
     * 获取合法的进度状态
     */
    getValidProgressStatus(status) {
      // Element UI el-progress 的有效status值
      const validStatuses = ['success', 'exception', 'warning']
      
      if (!status || status === '') {
        return undefined
      }
      
      // 映射常见状态
      const statusMap = {
        'error': 'exception',
        'failed': 'exception',
        'complete': 'success',
        'finished': 'success',
        'running': undefined,
        'processing': undefined,
        'in_progress': undefined,
        'in-progress': undefined
      }
      
      const mappedStatus = statusMap[status] || status
      
      // 如果是有效状态则返回，否则返回undefined
      return validStatuses.includes(mappedStatus) ? mappedStatus : undefined
    },

    /**
     * 处理分析错误
     */
    handleAnalysisError(error) {
      this.isLoading = false
      this.isStreaming = false

      let userMessage = error.message
      let details = ''
      let suggestions = []

      if (error.message.includes('401')) {
        userMessage = '登录已过期'
        details = '请刷新页面重新登录'
        suggestions = ['刷新页面重新登录', '检查token是否有效']
      } else if (error.message.includes('404')) {
        userMessage = 'API端点不存在'
        details = '请检查后端服务配置'
        suggestions = ['检查后端服务是否启动', '确认API路径是否正确']
      } else if (error.message.includes('413')) {
        userMessage = '请求数据过大'
        details = '请减少输入内容或使用URL分析'
        suggestions = ['使用URL分析替代长文本', '拆分文本为多个部分分析']
      } else if (error.message.includes('timeout') || error.message.includes('超时')) {
        userMessage = '分析超时'
        details = '可能是AI服务响应慢，请稍后重试'
        suggestions = ['稍后重试', '减少输入内容长度', '使用更快的模型']
      } else if (error.message.includes('Failed to fetch') || error.message.includes('网络连接')) {
        userMessage = '网络连接失败'
        details = '请检查网络连接和后端服务状态'
        suggestions = ['检查网络连接', '确认后端服务地址', '尝试重新连接']
      } else if (error.message.includes('AbortError')) {
        userMessage = '请求被取消'
        details = '可能是操作超时或用户取消'
        suggestions = ['重新开始分析', '检查网络稳定性']
      } else if (error.message.includes('rate limit') || error.message.includes('限流')) {
        userMessage = 'API调用频率限制'
        details = '已达到API调用频率限制'
        suggestions = ['稍后重试', '减少请求频率', '检查API配额']
      }

      this.errorMessage = userMessage
      this.errorDetails = details
      this.suggestions = suggestions
      this.showErrorDialog = true

      // 更新进度状态为错误
      this.progress = {
        ...this.progress,
        status: 'exception',  // 设置有效的错误状态
        message: userMessage
      }

      this.$message.error(`分析失败: ${userMessage}`)
    },

    /**
     * 重试分析
     */
    retryAnalysis() {
      this.showErrorDialog = false
      if (this.inputValue) {
        this.handleStartAnalysis({
          type: this.inputType,
          value: this.inputValue
        })
      }
    },

    /**
     * 处理流式数据 - 核心修复：正确处理边数据
     */
    handleStreamData(data) {
      console.log('📨 收到流式数据:', data.type, data)

      switch (data.type) {
        case 'content_block_delta':
          // 内容片段，收集起来用于最终解析
          break

        case 'node':
          this.handleNodeData(data.node)
          break

        case 'edge':
          this.handleEdgeData(data.edge)
          break

        case 'progress':
          this.handleProgress(data)
          break

        case 'complete':
          this.handleComplete(data)
          break

        case 'error':
          this.handleStreamError(data)
          break

        default:
          console.log('📨 未知事件类型:', data.type, data)
      }
    },

    /**
     * 处理节点数据 - 修复：确保节点正确添加到可视化
     */
    handleNodeData(nodeData) {
      if (!nodeData || !nodeData.id) {
        console.warn('无效的节点数据:', nodeData)
        return
      }

      console.log('✅ 收到节点:', nodeData.id, nodeData.type)

      // 添加到原始数据
      if (!this.rawNodes.find(n => n.id === nodeData.id)) {
        this.rawNodes.push(nodeData)
      }

      // 准备可视化节点（添加位置信息和data属性）
      const visualizationNode = {
        ...nodeData,
        // 确保有data属性,如果没有则创建
        data: nodeData.data || {
          label: nodeData.label || nodeData.id,
          description: nodeData.description || '',
          technique_id: nodeData.technique_id || '',
          tactic: nodeData.tactic || '',
          cve_id: nodeData.cve_id || '',
          ip_address: nodeData.ip_address || '',
          domain: nodeData.domain || ''
        },
        position: {
          x: 100 + (this.visualizationNodes.length % 10) * 250,
          y: 100 + Math.floor(this.visualizationNodes.length / 10) * 180
        }
      }

      // 添加到可视化数据
      const existingIndex = this.visualizationNodes.findIndex(n => n.id === nodeData.id)
      if (existingIndex === -1) {
        this.visualizationNodes.push(visualizationNode)

        // 如果有动画，添加淡入效果
        if (this.enableAnimations && this.streamingRender) {
          setTimeout(() => {
            const node = this.visualizationNodes.find(n => n.id === nodeData.id)
            if (node) {
              // 这里可以添加动画效果，如果需要的话
            }
          }, 50)
        }
      } else {
        // 更新现有节点
        this.visualizationNodes[existingIndex] = visualizationNode
      }

      // 触发更新
      this.visualizationNodes = [...this.visualizationNodes]
    },

    /**
     * 处理边数据 - 核心修复：确保边正确显示
     */
    handleEdgeData(edgeData) {
      if (!edgeData || !edgeData.source || !edgeData.target) {
        console.warn('无效的边数据:', edgeData)
        return
      }

      console.log('✅ 收到边:', edgeData.source, '→', edgeData.target)

      const edgeId = edgeData.id || `edge-${edgeData.source}-${edgeData.target}`

      // 检查源节点和目标节点是否存在
      const sourceExists = this.visualizationNodes.some(n => n.id === edgeData.source)
      const targetExists = this.visualizationNodes.some(n => n.id === edgeData.target)

      if (!sourceExists || !targetExists) {
        console.log('⏳ 等待节点加载:', edgeData.source, edgeData.target)
        // 如果节点不存在，先保存到原始数据，稍后处理
        if (!this.rawEdges.find(e => e.id === edgeId)) {
          this.rawEdges.push({
            ...edgeData,
            id: edgeId
          })
        }
        return
      }

      // 添加到原始数据
      if (!this.rawEdges.find(e => e.id === edgeId)) {
        this.rawEdges.push({
          ...edgeData,
          id: edgeId
        })
      }

      // 添加到可视化数据
      const visualizationEdge = {
        ...edgeData,
        id: edgeId,
        type: 'floating'
      }

      const existingIndex = this.visualizationEdges.findIndex(e => e.id === edgeId)
      if (existingIndex === -1) {
        this.visualizationEdges.push(visualizationEdge)

        // 如果有动画，添加高亮效果
        if (this.enableAnimations && this.streamingRender) {
          setTimeout(() => {
            // 这里可以添加边动画效果，如果需要的话
          }, 100)
        }
      } else {
        // 更新现有边
        this.visualizationEdges[existingIndex] = visualizationEdge
      }

      // 触发更新
      this.visualizationEdges = [...this.visualizationEdges]
    },

    /**
     * 处理进度更新 - 修复：确保status值是Element UI接受的
     */
    handleProgress(data) {
      this.progress = {
        stage: data.stage || '',
        message: data.message || '',
        percentage: data.percentage || 0,
        status: this.getValidProgressStatus(data.status)  // 使用辅助函数确保有效
      }
    },

    /**
     * 处理流式错误
     */
    handleStreamError(data) {
      console.error('❌ 流式错误:', data.error)
      this.handleAnalysisError(new Error(data.error || '流式处理错误'))
    },

    /**
     * 处理完成事件 - 修复：最终解析并确保所有边都显示
     */
    handleComplete(data) {
      console.log('🏁 流式分析完成', data)

      this.isLoading = false
      this.isStreaming = false

      // 计算分析时间
      if (this.analysisStartTime) {
        this.analysisTime = ((Date.now() - this.analysisStartTime) / 1000).toFixed(1)
      }

      // 确保所有边都显示
      this.processAllEdges()

      // 显示完成消息
      const nodeCount = this.visualizationNodes.length
      const edgeCount = this.visualizationEdges.length
      this.$message.success(`分析完成！生成 ${nodeCount} 个节点，${edgeCount} 条边`)

      // 更新进度 - 修复：设置有效的status值
      this.progress = {
        stage: 'complete',
        message: '分析完成',
        percentage: 100,
        status: 'success'  // 有效的Element UI状态值
      }

      // 保存分析结果
      this.saveAnalysisResult()
    },

    /**
     * 处理所有边 - 确保所有边都正确显示
     */
    processAllEdges() {
      console.log('🔗 处理所有边，原始边数量:', this.rawEdges.length)

      this.rawEdges.forEach(edgeData => {
        const edgeId = edgeData.id || `edge-${edgeData.source}-${edgeData.target}`

        // 检查是否已经在可视化数据中
        const existingEdge = this.visualizationEdges.find(e => e.id === edgeId)
        if (existingEdge) {
          return // 已经存在
        }

        // 检查节点是否存在
        const sourceExists = this.visualizationNodes.some(n => n.id === edgeData.source)
        const targetExists = this.visualizationNodes.some(n => n.id === edgeData.target)

        if (sourceExists && targetExists) {
          // 添加到可视化数据
          this.visualizationEdges.push({
            ...edgeData,
            id: edgeId,
            type: 'floating'
          })
          console.log('✅ 添加边:', edgeData.source, '→', edgeData.target)
        } else {
          console.log('❌ 无法添加边，节点不存在:', edgeData.source, edgeData.target)
        }
      })

      // 触发更新
      this.visualizationEdges = [...this.visualizationEdges]
      console.log('最终边数量:', this.visualizationEdges.length)
    },

    /**
     * 清理分析结果
     */
    handleClearAnalysis() {
      this.resetAnalysisState()
      this.$message.success('已清理分析结果')
    },

    /**
     * 处理节点点击
     */
    handleNodeClick(node) {
      this.selectedNode = node
      this.selectedEdge = null
      this.showNodeDetails = true
      this.showEdgeDetails = false
    },

    /**
     * 处理边点击
     */
    handleEdgeClick(edge) {
      this.selectedEdge = edge
      this.selectedNode = null
      this.showEdgeDetails = true
      this.showNodeDetails = false
    },

    /**
     * 处理节点详情关闭
     */
    handleNodeDetailsClose() {
      this.selectedNode = null
      this.showNodeDetails = false
    },

    /**
     * 处理边详情关闭
     */
    handleEdgeDetailsClose() {
      this.selectedEdge = null
      this.showEdgeDetails = false
    },

    /**
     * 处理节点更新
     */
    handleNodesUpdated(nodes) {
      this.visualizationNodes = nodes
    },

    /**
     * 处理边更新
     */
    handleEdgesUpdated(edges) {
      this.visualizationEdges = edges
    },

    /**
     * 处理导出
     */
    async handleExport(format) {
      if (!this.$refs.flowViz) {
        this.$message.warning('请先进行分析生成图表')
        return
      }

      try {
        await this.$refs.flowViz.exportDiagram(format)
        this.$message.success(`已导出${format.toUpperCase()}格式`)
      } catch (error) {
        console.error('导出失败:', error)
        this.$message.error(`导出失败: ${error.message}`)
      }
    },

    /**
     * 测试连接
     */
    async handleTestConnection() {
      this.$message.info('正在测试连接...')

      try {
        const result = await flowvizApi.quickTest()
        this.$message.success('连接测试成功: ' + (result.data?.status || '正常'))
      } catch (error) {
        this.$message.error('连接测试失败: ' + error.message)
      }
    },

    /**
     * 保存分析结果
     */
    saveAnalysisResult() {
      const result = {
        id: `flow-${Date.now()}`,
        title: `分析结果 ${new Date().toLocaleString()}`,
        inputType: this.inputType,
        inputValue: this.inputValue.substring(0, 100) + (this.inputValue.length > 100 ? '...' : ''),
        nodes: this.visualizationNodes,
        edges: this.visualizationEdges,
        provider: this.provider,
        model: this.model,
        analysisTime: this.analysisTime,
        createdAt: new Date().toISOString()
      }

      // 保存到localStorage
      let savedFlows = JSON.parse(localStorage.getItem('flowviz_saved_flows') || '[]')
      savedFlows.unshift(result)
      if (savedFlows.length > 50) {
        savedFlows = savedFlows.slice(0, 50)
      }
      localStorage.setItem('flowviz_saved_flows', JSON.stringify(savedFlows))

      console.log('💾 分析结果已保存')
    }
  }
}
</script>

<style scoped>
.flow-analysis {
  height: 100%;
  width: 100%;
  background: #ffffff;
}

.el-container {
  height: 100%;
}

/* 左侧控制面板 */
.control-aside {
  background: white;
  border-right: 1px solid #ebeef5;
  padding: 20px;
  overflow-y: auto;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
}

/* 主内容区域 */
.flow-main {
  padding: 0;
  position: relative;
  overflow: hidden;
  background: #ffffff;
}

/* 空状态页面 */
.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  z-index: 10;
  width: 90%;
  max-width: 800px;
}

.empty-state-content {
  background: white;
  padding: 40px;
  border-radius: 12px;
  border: 1px solid #ebeef5;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.empty-icon {
  font-size: 64px;
  color: #409EFF;
  margin-bottom: 20px;
  opacity: 0.7;
}

.empty-state h3 {
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 16px;
}

.description {
  font-size: 16px;
  color: #606266;
  line-height: 1.6;
  margin-bottom: 30px;
}

.mode-tips {
  display: flex;
  gap: 30px;
  margin-top: 30px;
  justify-content: center;
}

.mode-tip {
  flex: 1;
  max-width: 300px;
  padding: 20px;
  border-radius: 8px;
  text-align: center;
  border: 2px solid #ebeef5;
  transition: all 0.3s;
}

.mode-tip:hover {
  border-color: #409EFF;
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.mode-tip i {
  font-size: 36px;
  margin-bottom: 15px;
  display: block;
}

.url-tip i {
  color: #409EFF;
}

.text-tip i {
  color: #67C23A;
}

.mode-tip h4 {
  margin: 0 0 10px 0;
  font-size: 18px;
  color: #303133;
}

.mode-tip p {
  margin: 0;
  font-size: 14px;
  color: #606266;
  line-height: 1.5;
}

/* 流式处理指示器 */
.streaming-indicator {
  position: absolute;
  top: 20px;
  right: 20px;
  display: flex;
  align-items: center;
  background: white;
  padding: 10px 20px;
  border-radius: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border: 1px solid #ebeef5;
  z-index: 100;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(64, 158, 255, 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(64, 158, 255, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(64, 158, 255, 0);
  }
}

.spinner {
  width: 40px;
  height: 20px;
  margin-right: 10px;
  text-align: center;
}

.spinner > div {
  width: 8px;
  height: 8px;
  background-color: #409EFF;
  border-radius: 100%;
  display: inline-block;
  animation: bounce 1.4s infinite ease-in-out both;
}

.spinner .bounce1 {
  animation-delay: -0.32s;
}

.spinner .bounce2 {
  animation-delay: -0.16s;
}

@keyframes bounce {
  0%, 80%, 100% {
    transform: scale(0);
  }
  40% {
    transform: scale(1.0);
  }
}

.streaming-indicator span {
  font-size: 14px;
  color: #409EFF;
  font-weight: 500;
  margin-right: 10px;
}

.streaming-indicator .stats {
  font-size: 12px;
  color: #909399;
  background: #f5f7fa;
  padding: 2px 8px;
  border-radius: 10px;
  margin-left: 5px;
}

/* 统计信息 */
.stats-info {
  position: absolute;
  bottom: 20px;
  left: 20px;
  display: flex;
  gap: 10px;
  z-index: 100;
}

/* 错误对话框 */
.error-content {
  text-align: center;
  padding: 20px;
}

.suggestions {
  margin-top: 20px;
  text-align: left;
}

.suggestions h4 {
  margin-bottom: 10px;
  color: #303133;
}

.suggestions ul {
  margin: 0;
  padding-left: 20px;
  color: #606266;
}

.suggestions li {
  margin-bottom: 5px;
  font-size: 14px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .control-aside {
    width: 280px !important;
  }

  .mode-tips {
    flex-direction: column;
    align-items: center;
  }

  .mode-tip {
    max-width: 100%;
    width: 100%;
  }
}

@media (max-width: 768px) {
  .el-container {
    flex-direction: column;
  }

  .control-aside {
    width: 100% !important;
    height: auto !important;
    max-height: 400px;
  }

  .empty-state h3 {
    font-size: 24px;
  }

  .description {
    font-size: 14px;
  }

  .streaming-indicator {
    top: 10px;
    right: 10px;
    padding: 6px 12px;
    flex-wrap: wrap;
  }

  .stats-info {
    bottom: 10px;
    left: 10px;
    flex-wrap: wrap;
  }
}
</style>