<template>
  <div ref="flowWrapper" class="flow-visualization-wrapper">
    <!-- 工具栏切换按钮 -->
    <el-button
      class="toolbar-toggle"
      size="small"
      circle
      @click="showToolbar = !showToolbar"
    >
      <i :class="showToolbar ? 'el-icon-arrow-up' : 'el-icon-setting'" />
    </el-button>

    <!-- 工具栏 -->
    <div v-show="showToolbar" class="flow-toolbar">
      <div class="toolbar-group">
        <el-button-group>
          <el-tooltip content="放大" placement="bottom">
            <el-button size="small" @click="zoomIn">
              <i class="el-icon-zoom-in" />
            </el-button>
          </el-tooltip>
          <el-tooltip content="缩小" placement="bottom">
            <el-button size="small" @click="zoomOut">
              <i class="el-icon-zoom-out" />
            </el-button>
          </el-tooltip>
          <el-tooltip content="适应视图" placement="bottom">
            <el-button size="small" @click="fitView">
              <i class="el-icon-view" />
            </el-button>
          </el-tooltip>
          <el-tooltip content="重置" placement="bottom">
            <el-button size="small" @click="resetView">
              <i class="el-icon-refresh" />
            </el-button>
          </el-tooltip>
        </el-button-group>
      </div>

      <div class="toolbar-group">
        <el-select
          v-model="edgeStyle"
          size="small"
          placeholder="连线样式"
          style="width: 100px;"
        >
          <el-option label="平滑曲线" value="smooth" />
          <el-option label="直线" value="straight" />
          <el-option label="阶梯线" value="step" />
          <el-option label="虚线" value="dashed" />
        </el-select>

        <el-color-picker
          v-model="edgeColor"
          size="small"
          :predefine="predefineColors"
          style="margin-left: 10px;"
        />

        <el-select
          v-model="layoutDirection"
          size="small"
          placeholder="布局方向"
          style="width: 100px; margin-left: 10px;"
          @change="applyLayout"
        >
          <el-option label="从上到下" value="TB" />
          <el-option label="从左到右" value="LR" />
          <el-option label="从右到左" value="RL" />
          <el-option label="从下到上" value="BT" />
          <el-option label="放射状" value="radial" />
        </el-select>
      </div>

      <div class="toolbar-group">
        <el-button-group>
          <el-tooltip content="显示/隐藏网格" placement="bottom">
            <el-button
              size="small"
              :type="showGrid ? 'primary' : ''"
              @click="toggleGrid"
            >
              <i class="el-icon-set-up" />
            </el-button>
          </el-tooltip>
          <el-tooltip content="显示/隐藏节点标签" placement="bottom">
            <el-button
              size="small"
              :type="showLabels ? 'primary' : ''"
              @click="toggleLabels"
            >
              <i class="el-icon-tickets" />
            </el-button>
          </el-tooltip>
          <el-tooltip content="显示/隐藏连线标签" placement="bottom">
            <el-button
              size="small"
              :type="showEdgeLabels ? 'primary' : ''"
              @click="toggleEdgeLabels"
            >
              <i class="el-icon-connection" />
            </el-button>
          </el-tooltip>
          <el-tooltip content="显示/隐藏节点详情" placement="bottom">
            <el-button
              size="small"
              :type="showNodeDetails ? 'primary' : ''"
              @click="toggleNodeDetails"
            >
              <i class="el-icon-info" />
            </el-button>
          </el-tooltip>
        </el-button-group>
      </div>
    </div>

    <!-- 滚动提示 -->
    <div v-if="showScrollHint" class="scroll-hint">
      <i class="el-icon-rank" />
      <span>可拖动画布查看完整图表</span>
    </div>

    <!-- 空状态 -->
    <div v-if="!hasData && !isStreaming" class="empty-state">
      <div class="empty-state-content">
        <i class="el-icon-data-analysis empty-icon" />
        <h3>攻击流程图</h3>
        <p>分析结果将在此可视化展示</p>
        <p class="hint">支持拖拽画布、缩放视图、点击查看详情</p>
      </div>
    </div>

    <!-- 主画布 -->
    <div
      ref="flowCanvas"
      class="flow-canvas"
      :style="canvasStyle"
      @mousedown="handleCanvasMouseDown"
      @wheel.prevent="handleWheel"
      @mouseenter="handleCanvasMouseEnter"
      @mouseleave="handleCanvasMouseLeave"
    >
      <!-- 网格背景 -->
      <div v-if="showGrid" class="grid-background" :style="gridStyle" />

      <!-- 节点容器 -->
      <div
        v-for="node in layoutedNodes"
        :key="node.id"
        class="flow-node"
        :class="[
          `node-type-${node.type}`,
          { 'node-selected': selectedNodeId === node.id },
          { 'node-highlighted': highlightedNodeIds.has(node.id) }
        ]"
        :style="getNodeStyle(node)"
        @mousedown.stop="handleNodeMouseDown(node, $event)"
        @click.stop="handleNodeClick(node)"
        @mouseenter="handleNodeMouseEnter(node)"
        @mouseleave="handleNodeMouseLeave(node)"
      >
        <!-- 节点连接点 -->
        <div
          class="node-handle node-handle-top"
          @mousedown.stop="handleHandleMouseDown(node, 'top')"
        />
        <div
          class="node-handle node-handle-bottom"
          @mousedown.stop="handleHandleMouseDown(node, 'bottom')"
        />
        <div
          class="node-handle node-handle-left"
          @mousedown.stop="handleHandleMouseDown(node, 'left')"
        />
        <div
          class="node-handle node-handle-right"
          @mousedown.stop="handleHandleMouseDown(node, 'right')"
        />

        <!-- 节点头部 -->
        <div class="node-header" :style="{ background: getNodeHeaderColor(node.type) }">
          <div class="node-title-wrapper">
            <div class="node-title" :title="getNodeTitle(node)">
              {{ truncateText(getNodeTitle(node), 25) }}
            </div>
            <div class="node-type-badge">
              {{ getNodeTypeLabel(node.type) }}
            </div>
          </div>
          <div class="node-actions">
            <i
              v-if="!node.favorite"
              class="el-icon-star-off"
              @click.stop="toggleFavorite(node)"
            />
            <i
              v-else
              class="el-icon-star-on"
              style="color: #F56C6C;"
              @click.stop="toggleFavorite(node)"
            />
          </div>
        </div>

        <!-- 节点内容 -->
        <div class="node-content">
          <!-- MITRE ATT&CK信息 -->
          <div v-if="node.type === 'action'" class="mitre-info">
            <div v-if="node.data.technique_id" class="mitre-technique">
              <el-tag size="small" type="danger" class="technique-tag">
                {{ node.data.technique_id }}
              </el-tag>
              <span v-if="node.data.tactic" class="tactic-text">
                {{ node.data.tactic }}
              </span>
            </div>
          </div>

          <!-- 描述 -->
          <div
            v-if="showNodeDetails && node.data.description"
            class="node-description"
            :title="node.data.description"
          >
            {{ truncateText(node.data.description, 60) }}
          </div>

          <!-- 技术信息 -->
          <div v-if="showNodeDetails && hasTechnicalInfo(node)" class="node-technical">
            <div v-if="node.data.cve_id" class="technical-item">
              <span class="technical-label">CVE:</span>
              <el-tag size="small" type="danger" class="technical-value">
                {{ node.data.cve_id }}
              </el-tag>
            </div>
            <div v-if="node.data.ip_address" class="technical-item">
              <span class="technical-label">IP:</span>
              <span class="technical-value code">{{ node.data.ip_address }}</span>
            </div>
            <div v-if="node.data.domain" class="technical-item">
              <span class="technical-label">域名:</span>
              <span class="technical-value code">{{ node.data.domain }}</span>
            </div>
            <div v-if="node.data.hash" class="technical-item">
              <span class="technical-label">哈希:</span>
              <span class="technical-value code">{{ truncateText(node.data.hash, 10) }}</span>
            </div>
          </div>

          <!-- 置信度 -->
          <div v-if="node.data.confidence" class="node-confidence">
            <el-tag
              size="small"
              :type="getConfidenceType(node.data.confidence)"
              class="confidence-tag"
            >
              {{ getConfidenceText(node.data.confidence) }}
            </el-tag>
          </div>

          <!-- 流式处理指示器 -->
          <div v-if="isStreaming && node._isNew" class="node-streaming-indicator">
            <i class="el-icon-loading" />
            <span>新节点</span>
          </div>
        </div>

        <!-- 节点底部 -->
        <div class="node-footer">
          <div v-if="node.data.source_excerpt" class="source-indicator" title="有来源引用">
            <i class="el-icon-document" />
          </div>
          <div v-if="connectedEdgesCount(node.id) > 0" class="connections-count">
            <i class="el-icon-connection" />
            {{ connectedEdgesCount(node.id) }}
          </div>
        </div>
      </div>

      <!-- SVG连线容器 -->
      <svg class="edges-container" :style="svgContainerStyle">
        <defs>
          <!-- 箭头标记 -->
          <marker
            id="arrowhead-default"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
            :fill="edgeColor"
          >
            <polygon points="0 0, 10 3.5, 0 7" />
          </marker>

          <!-- 其他箭头标记 -->
          <marker
            id="arrowhead-blue"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
            fill="#409EFF"
          >
            <polygon points="0 0, 10 3.5, 0 7" />
          </marker>

          <!-- 虚线样式 -->
          <pattern id="dashPattern" patternUnits="userSpaceOnUse" width="8" height="8">
            <path d="M 0,4 l 8,0" stroke-width="2" stroke="#409EFF" fill="none" />
          </pattern>

          <!-- 渐变样式 -->
          <linearGradient id="edge-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" :stop-color="edgeColor" stop-opacity="0.8" />
            <stop offset="100%" :stop-color="edgeColor" stop-opacity="0.4" />
          </linearGradient>

          <!-- 动画边标记 -->
          <marker
            id="arrowhead-animated"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
            fill="#F56C6C"
          >
            <polygon points="0 0, 10 3.5, 0 7" />
          </marker>
        </defs>

        <!-- 绘制所有边 -->
        <g v-for="edge in filteredEdges" :key="edge.id">
          <!-- 连线路径 -->
          <path
            :d="getEdgePath(edge)"
            fill="none"
            :stroke="getEdgeColor(edge)"
            :stroke-width="getEdgeWidth(edge)"
            :stroke-dasharray="getEdgeDashArray(edge)"
            :marker-end="getEdgeMarker(edge)"
            class="flow-edge"
            :class="{
              'edge-highlighted': highlightedEdgeIds.has(edge.id),
              'edge-selected': selectedEdgeId === edge.id,
              'edge-animated': edge.animated && enableAnimations
            }"
            @click.stop="handleEdgeClick(edge)"
            @mouseenter="handleEdgeMouseEnter(edge)"
            @mouseleave="handleEdgeMouseLeave(edge)"
          />

          <!-- 连线标签 -->
          <g v-if="showEdgeLabels && (edge.label || edge.relation)" class="edge-label-container">
            <!-- 计算标签位置 -->
            <rect
              :x="getEdgeLabelPosition(edge).x - getLabelWidth(edge) / 2"
              :y="getEdgeLabelPosition(edge).y - 14"
              :width="getLabelWidth(edge)"
              :height="28"
              rx="6"
              fill="white"
              stroke="#DCDFE6"
              stroke-width="1"
              class="edge-label-bg"
              style="filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.15));"
            />
            <text
              :x="getEdgeLabelPosition(edge).x"
              :y="getEdgeLabelPosition(edge).y + 2"
              text-anchor="middle"
              dominant-baseline="middle"
              font-size="11"
              font-weight="600"
              :fill="getEdgeLabelColor(edge)"
              class="edge-label"
            >
              {{ getEdgeLabelText(edge) }}
            </text>
          </g>
        </g>

        <!-- 正在绘制的连线 -->
        <path
          v-if="drawingEdge"
          :d="drawingEdgePath"
          fill="none"
          stroke="#909399"
          stroke-width="1.5"
          stroke-dasharray="4 4"
          class="drawing-edge"
        />
      </svg>

      <!-- 高亮相关节点的连线 -->
      <svg v-if="highlightedNodeIds.size > 0" class="highlight-edges-container" :style="svgContainerStyle">
        <g v-for="edge in highlightEdges" :key="`highlight-${edge.id}`">
          <path
            :d="getEdgePath(edge)"
            fill="none"
            stroke="#F56C6C"
            stroke-width="3"
            opacity="0.6"
            class="highlight-edge"
          />
        </g>
      </svg>
    </div>

    <!-- 流式处理指示器 -->
    <div v-if="isStreaming" class="streaming-indicator">
      <div class="spinner">
        <div class="bounce1" />
        <div class="bounce2" />
        <div class="bounce3" />
      </div>
      <span>正在生成节点...</span>
      <span class="node-count">{{ layoutedNodes.length }} 节点</span>
      <span class="edge-count">{{ filteredEdges.length }} 边</span>
    </div>

    <!-- 统计信息 -->
    <div v-if="hasData && !isStreaming" class="stats-panel">
      <div class="stats-item">
        <span class="stats-label">节点:</span>
        <span class="stats-value">{{ layoutedNodes.length }}</span>
      </div>
      <div class="stats-item">
        <span class="stats-label">边:</span>
        <span class="stats-value">{{ filteredEdges.length }}</span>
      </div>
      <div class="stats-item">
        <span class="stats-label">攻击行动:</span>
        <span class="stats-value">{{ actionNodeCount }}</span>
      </div>
      <div class="stats-item">
        <span class="stats-label">方向:</span>
        <span class="stats-value">{{ getLayoutDirectionLabel(layoutDirection) }}</span>
      </div>
      <div class="stats-item">
        <span class="stats-label">画布尺寸:</span>
        <span class="stats-value">{{ canvasBounds.width }}×{{ canvasBounds.height }}</span>
      </div>
    </div>
  </div>
</template>

<script>
// 使用dagre进行自动布局
import dagre from 'dagre'

export default {
  name: 'FlowVisualization',
  props: {
    nodes: {
      type: Array,
      default: () => []
    },
    edges: {
      type: Array,
      default: () => []
    },
    isStreaming: {
      type: Boolean,
      default: false
    },
    streamingRender: {
      type: Boolean,
      default: true
    },
    enableAnimations: {
      type: Boolean,
      default: true
    }
  },
  data() {
    return {
      showToolbar: false, // 默认隐藏工具栏
      layoutedNodes: [],
      selectedNodeId: null,
      selectedEdgeId: null,
      highlightedNodeIds: new Set(),
      highlightedEdgeIds: new Set(),
      zoom: 1.0,
      pan: { x: 0, y: 0 },
      isPanning: false,
      panStart: { x: 0, y: 0 },
      edgeStyle: 'straight',
      edgeColor: '#409EFF',
      showGrid: true,
      showLabels: true,
      showEdgeLabels: true,
      showNodeDetails: true,
      showScrollHint: false,
      layoutDirection: 'TB',
      drawingEdge: null,
      drawingEdgePath: '',
      nodeWidth: 280,
      nodeHeight: 160,
      // 动态画布边界
      canvasBounds: {
        minX: 0,
        minY: 0,
        maxX: 3000,
        maxY: 2000,
        width: 3000,
        height: 2000
      },
      nodeTypes: {
        action: { color: '#409EFF', label: '攻击行动', gradient: 'linear-gradient(135deg, #409EFF 0%, #2979ff 100%)' },
        tool: { color: '#67C23A', label: '工具', gradient: 'linear-gradient(135deg, #67C23A 0%, #5daf34 100%)' },
        malware: { color: '#F56C6C', label: '恶意软件', gradient: 'linear-gradient(135deg, #F56C6C 0%, #f45151 100%)' },
        asset: { color: '#E6A23C', label: '资产', gradient: 'linear-gradient(135deg, #E6A23C 0%, #e29419 100%)' },
        infrastructure: { color: '#06b6d4', label: '基础设施', gradient: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)' },
        vulnerability: { color: '#f43f5e', label: '漏洞', gradient: 'linear-gradient(135deg, #f43f5e 0%, #e11d48 100%)' },
        file: { color: '#8b5cf6', label: '文件', gradient: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)' },
        process: { color: '#10b981', label: '进程', gradient: 'linear-gradient(135deg, #10b981 0%, #059669 100%)' },
        network: { color: '#3b82f6', label: '网络', gradient: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)' },
        registry: { color: '#8b5cf6', label: '注册表', gradient: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)' },
        AND_operator: { color: '#909399', label: 'AND门', gradient: 'linear-gradient(135deg, #909399 0%, #7a7c80 100%)' },
        OR_operator: { color: '#909399', label: 'OR门', gradient: 'linear-gradient(135deg, #909399 0%, #7a7c80 100%)' },
        default: { color: '#909399', label: '节点', gradient: 'linear-gradient(135deg, #909399 0%, #7a7c80 100%)' }
      },
      predefineColors: [
        '#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#909399',
        '#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#06b6d4',
        '#8b5cf6', '#f43f5e', '#84cc16', '#14b8a6', '#a855f7'
      ]
    }
  },

  computed: {
    hasData() {
      return this.layoutedNodes.length > 0 || this.edges.length > 0
    },

    canvasStyle() {
      return {
        transform: `translate(${this.pan.x}px, ${this.pan.y}px) scale(${this.zoom})`,
        transformOrigin: '0 0',
        width: `${this.canvasBounds.width}px`,
        height: `${this.canvasBounds.height}px`
      }
    },

    gridStyle() {
      return {
        width: `${this.canvasBounds.width}px`,
        height: `${this.canvasBounds.height}px`
      }
    },

    svgContainerStyle() {
      return {
        width: `${this.canvasBounds.width}px`,
        height: `${this.canvasBounds.height}px`
      }
    },

    filteredEdges() {
      return this.edges
    },

    highlightEdges() {
      return this.edges.filter(edge => {
        return this.highlightedNodeIds.has(edge.source) ||
               this.highlightedNodeIds.has(edge.target)
      })
    },

    nodeTypeCount() {
      const types = new Set(this.layoutedNodes.map(n => n.type))
      return types.size
    },

    actionNodeCount() {
      return this.layoutedNodes.filter(n => n.type === 'action').length
    }
  },

  watch: {
    nodes: {
      immediate: true,
      handler(newNodes) {
        console.log('节点更新:', newNodes.length, '个节点')
        if (newNodes.length > 0) {
          this.updateLayout(newNodes, this.edges)
        } else {
          this.layoutedNodes = []
          this.resetCanvasBounds()
        }
      }
    },

    edges: {
      handler(newEdges) {
        console.log('边更新:', newEdges.length, '条边')
        if (this.layoutedNodes.length > 0) {
          this.applyLayout(this.layoutedNodes, newEdges)
        }
      }
    },

    layoutDirection() {
      if (this.layoutedNodes.length > 0) {
        this.applyLayout(this.layoutedNodes, this.edges)
      }
    }
  },

  mounted() {
    this.setupEventListeners()
    this.updateCanvasSize()
    window.addEventListener('resize', this.updateCanvasSize)

    // 初始化布局
    if (this.nodes.length > 0) {
      this.updateLayout(this.nodes, this.edges)
    }
  },

  beforeDestroy() {
    this.removeEventListeners()
    window.removeEventListener('resize', this.updateCanvasSize)
  },

  methods: {
    /**
     * 获取节点样式
     */
    getNodeStyle(node) {
      return {
        left: node.position.x + 'px',
        top: node.position.y + 'px',
        width: (node.width || this.nodeWidth) + 'px',
        minHeight: (node.height || this.nodeHeight) + 'px',
        borderLeftColor: this.getNodeColor(node.type),
        opacity: node.style?.opacity || 1,
        transform: node.style?.transform || 'scale(1)',
        transition: node.style?.transition || 'all 0.3s ease'
      }
    },

    /**
     * 获取节点标题
     */
    getNodeTitle(node) {
      return node.data?.label || node.data?.name || node.data?.technique_id || node.id
    },

    /**
     * 获取边标签文本
     */
    getEdgeLabelText(edge) {
      return edge.label || edge.relation || edge.data?.label || '关联'
    },

    /**
     * 获取边标签颜色
     */
    getEdgeLabelColor(edge) {
      const label = this.getEdgeLabelText(edge).toLowerCase()
      if (label.includes('leads to') || label.includes('progresses')) {
        return '#409EFF'
      } else if (label.includes('uses') || label.includes('deploys')) {
        return '#67C23A'
      } else if (label.includes('targets') || label.includes('affects')) {
        return '#E6A23C'
      } else if (label.includes('exploits') || label.includes('creates')) {
        return '#F56C6C'
      } else if (label.includes('communicates') || label.includes('connects')) {
        return '#06b6d4'
      }
      return '#606266'
    },

    /**
     * 获取标签宽度
     */
    getLabelWidth(edge) {
      const text = this.getEdgeLabelText(edge)
      return Math.max(text.length * 7.5, 40)
    },

    /**
     * 设置事件监听器
     */
    setupEventListeners() {
      document.addEventListener('mousemove', this.handleMouseMove)
      document.addEventListener('mouseup', this.handleMouseUp)
      document.addEventListener('keydown', this.handleKeyDown)
    },

    removeEventListeners() {
      document.removeEventListener('mousemove', this.handleMouseMove)
      document.removeEventListener('mouseup', this.handleMouseUp)
      document.removeEventListener('keydown', this.handleKeyDown)
    },

    /**
     * 更新画布尺寸
     */
    updateCanvasSize() {
      if (this.$refs.flowWrapper) {
        const wrapper = this.$refs.flowWrapper
        this.$nextTick(() => {
          if (this.layoutedNodes.length > 0) {
            this.updateCanvasBounds()
          }
        })
      }
    },

    /**
     * 重置画布边界
     */
    resetCanvasBounds() {
      this.canvasBounds = {
        minX: 0,
        minY: 0,
        maxX: 3000,
        maxY: 2000,
        width: 3000,
        height: 2000
      }
    },

    /**
     * 更新画布边界（根据节点位置动态调整）
     */
    updateCanvasBounds() {
      if (this.layoutedNodes.length === 0) {
        this.resetCanvasBounds()
        return
      }

      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity

      this.layoutedNodes.forEach(node => {
        const nodeWidth = node.width || this.nodeWidth
        const nodeHeight = node.height || this.nodeHeight
        
        minX = Math.min(minX, node.position.x)
        minY = Math.min(minY, node.position.y)
        maxX = Math.max(maxX, node.position.x + nodeWidth)
        maxY = Math.max(maxY, node.position.y + nodeHeight)
      })

      // 添加边距
      const padding = 300
      minX = Math.min(minX, 0) - padding
      minY = Math.min(minY, 0) - padding
      maxX = Math.max(maxX, 1000) + padding
      maxY = Math.max(maxY, 800) + padding

      // 确保最小尺寸
      const minWidth = 3000
      const minHeight = 2000
      const width = Math.max(minWidth, maxX - minX)
      const height = Math.max(minHeight, maxY - minY)

      this.canvasBounds = {
        minX,
        minY,
        maxX: minX + width,
        maxY: minY + height,
        width,
        height
      }

      console.log('画布边界更新:', this.canvasBounds)
    },

    /**
     * 更新布局
     */
    updateLayout(nodes, edges) {
      console.log('更新布局，节点:', nodes.length, '边:', edges.length)

      // 深度复制节点，添加布局所需属性
      const nodesWithLayout = nodes.map(node => ({
        ...node,
        width: node.width || this.nodeWidth,
        height: node.height || this.nodeHeight,
        position: node.position || { x: 0, y: 0 }
      }))

      this.applyLayout(nodesWithLayout, edges)
    },

    /**
     * 应用自动布局 - 改进版本
     */
    applyLayout(nodes, edges) {
      if (nodes.length === 0) {
        this.layoutedNodes = []
        this.resetCanvasBounds()
        return
      }

      console.log('应用布局，节点:', nodes.length, '边:', edges.length, '方向:', this.layoutDirection)

      try {
        let layoutedNodes = []
        
        if (this.layoutDirection === 'radial') {
          layoutedNodes = this.applyRadialLayout(nodes, edges)
        } else {
          layoutedNodes = this.applyDagreLayout(nodes, edges)
        }

        this.layoutedNodes = layoutedNodes
        this.updateCanvasBounds()
        
        // 检查是否有重叠节点
        this.checkAndResolveOverlaps()
        
        // 自动适应视图
        this.$nextTick(() => {
          this.fitView()
        })

      } catch (error) {
        console.error('布局计算失败:', error)
        // 使用改进的网格布局
        this.layoutedNodes = nodes.map((node, index) => ({
          ...node,
          position: this.getBetterGridPosition(index, nodes.length)
        }))
        this.updateCanvasBounds()
      }
    },

    /**
     * 应用dagre布局
     */
    applyDagreLayout(nodes, edges) {
      const graph = new dagre.graphlib.Graph()
      graph.setDefaultEdgeLabel(() => ({}))

      // 设置图配置 - 优化布局参数
      graph.setGraph({
        rankdir: this.layoutDirection,
        nodesep: nodes.length > 30 ? 120 : 180,
        ranksep: nodes.length > 30 ? 150 : 250,
        marginx: 150,
        marginy: 150,
        edgesep: 80,
        ranker: nodes.length > 50 ? 'tight-tree' : 'longest-path'
      })

      // 添加节点
      nodes.forEach(node => {
        graph.setNode(node.id, {
          width: node.width || this.nodeWidth,
          height: node.height || this.nodeHeight,
          label: node.id
        })
      })

      // 添加边
      edges.forEach(edge => {
        if (edge.source && edge.target) {
          const sourceNode = nodes.find(n => n.id === edge.source)
          const targetNode = nodes.find(n => n.id === edge.target)

          if (sourceNode && targetNode) {
            graph.setEdge(edge.source, edge.target, {
              label: this.getEdgeLabelText(edge),
              minlen: 1
            })
          }
        }
      })

      // 计算布局
      dagre.layout(graph)

      // 更新节点位置
      return nodes.map(node => {
        const graphNode = graph.node(node.id)

        if (graphNode && graphNode.x && graphNode.y) {
          return {
            ...node,
            position: {
              x: graphNode.x - (node.width || this.nodeWidth) / 2,
              y: graphNode.y - (node.height || this.nodeHeight) / 2
            },
            width: node.width || this.nodeWidth,
            height: node.height || this.nodeHeight
          }
        }

        // 如果没有布局信息，使用网格布局
        return {
          ...node,
          position: this.getBetterGridPosition(nodes.indexOf(node), nodes.length),
          width: node.width || this.nodeWidth,
          height: node.height || this.nodeHeight
        }
      })
    },

    /**
     * 应用放射状布局
     */
    applyRadialLayout(nodes, edges) {
      const centerX = this.canvasBounds.width / 2
      const centerY = this.canvasBounds.height / 2
      const radius = Math.min(centerX, centerY) * 0.6
      
      // 尝试找到中心节点（连接最多的节点）
      let centerNodeId = nodes[0]?.id
      if (edges.length > 0) {
        const nodeConnections = {}
        edges.forEach(edge => {
          nodeConnections[edge.source] = (nodeConnections[edge.source] || 0) + 1
          nodeConnections[edge.target] = (nodeConnections[edge.target] || 0) + 1
        })
        
        const maxConnections = Math.max(...Object.values(nodeConnections))
        centerNodeId = Object.keys(nodeConnections).find(
          id => nodeConnections[id] === maxConnections
        ) || nodes[0]?.id
      }

      // 计算每层间距
      const levels = this.calculateRadialLevels(nodes, edges, centerNodeId)
      const angleStep = (2 * Math.PI) / nodes.length

      return nodes.map((node, index) => {
        let x, y
        
        if (node.id === centerNodeId) {
          x = centerX
          y = centerY
        } else {
          const level = levels[node.id] || 1
          const angle = index * angleStep
          const levelRadius = radius * (level * 0.3)
          
          x = centerX + Math.cos(angle) * levelRadius
          y = centerY + Math.sin(angle) * levelRadius
        }

        return {
          ...node,
          position: {
            x: x - (node.width || this.nodeWidth) / 2,
            y: y - (node.height || this.nodeHeight) / 2
          },
          width: node.width || this.nodeWidth,
          height: node.height || this.nodeHeight
        }
      })
    },

    /**
     * 计算放射状布局层级
     */
    calculateRadialLevels(nodes, edges, centerNodeId) {
      const levels = { [centerNodeId]: 0 }
      const visited = new Set([centerNodeId])
      const queue = [centerNodeId]
      
      while (queue.length > 0) {
        const currentNode = queue.shift()
        const currentLevel = levels[currentNode]
        
        // 查找与当前节点相连的节点
        const connectedNodes = edges
          .filter(edge => edge.source === currentNode || edge.target === currentNode)
          .map(edge => edge.source === currentNode ? edge.target : edge.source)
          .filter(nodeId => !visited.has(nodeId))
        
        connectedNodes.forEach(nodeId => {
          if (!visited.has(nodeId)) {
            visited.add(nodeId)
            levels[nodeId] = currentLevel + 1
            queue.push(nodeId)
          }
        })
      }
      
      // 为未连接的节点分配层级
      nodes.forEach(node => {
        if (!levels[node.id]) {
          levels[node.id] = Math.floor(Math.random() * 3) + 2
        }
      })
      
      return levels
    },

    /**
     * 获取更好的网格位置
     */
    getBetterGridPosition(index, totalNodes) {
      const cols = Math.ceil(Math.sqrt(totalNodes))
      const rows = Math.ceil(totalNodes / cols)
      
      const col = index % cols
      const row = Math.floor(index / cols)
      
      const padding = 100
      const nodeSpacingX = 350
      const nodeSpacingY = 250
      
      const startX = padding
      const startY = padding
      
      return {
        x: startX + col * nodeSpacingX,
        y: startY + row * nodeSpacingY
      }
    },

    /**
     * 检查并解决节点重叠
     */
    checkAndResolveOverlaps() {
      const nodes = this.layoutedNodes
      const overlapThreshold = 50
      let moved = false

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const nodeA = nodes[i]
          const nodeB = nodes[j]

          const aWidth = nodeA.width || this.nodeWidth
          const aHeight = nodeA.height || this.nodeHeight
          const bWidth = nodeB.width || this.nodeWidth
          const bHeight = nodeB.height || this.nodeHeight

          const overlapX = Math.max(0, Math.min(nodeA.position.x + aWidth, nodeB.position.x + bWidth) - Math.max(nodeA.position.x, nodeB.position.x))
          const overlapY = Math.max(0, Math.min(nodeA.position.y + aHeight, nodeB.position.y + bHeight) - Math.max(nodeA.position.y, nodeB.position.y))

          if (overlapX > 0 && overlapY > 0) {
            // 有重叠，移动节点B
            nodeB.position.x += aWidth + 50
            nodeB.position.y += 50
            moved = true
          }
        }
      }

      if (moved) {
        this.layoutedNodes = [...nodes]
        this.updateCanvasBounds()
      }
    },

    /**
     * 获取节点颜色
     */
    getNodeColor(nodeType) {
      return this.nodeTypes[nodeType]?.color || this.nodeTypes.default.color
    },

    /**
     * 获取节点头部颜色
     */
    getNodeHeaderColor(nodeType) {
      return this.nodeTypes[nodeType]?.gradient || this.nodeTypes.default.gradient
    },

    /**
     * 获取节点类型标签
     */
    getNodeTypeLabel(nodeType) {
      return this.nodeTypes[nodeType]?.label || nodeType
    },

    /**
     * 获取布局方向标签
     */
    getLayoutDirectionLabel(direction) {
      const map = {
        'TB': '上下',
        'LR': '左右',
        'RL': '右左',
        'BT': '下上',
        'radial': '放射状'
      }
      return map[direction] || direction
    },

    /**
     * 检查是否有技术信息
     */
    hasTechnicalInfo(node) {
      return node.data?.technique_id || node.data?.tactic || 
             node.data?.cve_id || node.data?.ip_address || 
             node.data?.domain || node.data?.hash
    },

    /**
     * 获取置信度类型
     */
    getConfidenceType(confidence) {
      const map = {
        'high': 'success',
        'medium': 'warning',
        'low': 'danger',
        'info': 'info'
      }
      return map[confidence] || 'info'
    },

    /**
     * 获取置信度文本
     */
    getConfidenceText(confidence) {
      const map = {
        'high': '高',
        'medium': '中',
        'low': '低',
        'info': '信息'
      }
      return map[confidence] || confidence
    },

    /**
     * 获取连接边数量
     */
    connectedEdgesCount(nodeId) {
      return this.edges.filter(edge =>
        edge.source === nodeId || edge.target === nodeId
      ).length
    },

    /**
     * 截断文本
     */
    truncateText(text, maxLength) {
      if (!text) return ''
      if (text.length <= maxLength) return text
      return text.substring(0, maxLength) + '...'
    },

    /**
     * 获取边路径 - 改进版本，确保边在画布内
     */
    getEdgePath(edge) {
      const sourceNode = this.layoutedNodes.find(n => n.id === edge.source)
      const targetNode = this.layoutedNodes.find(n => n.id === edge.target)

      if (!sourceNode || !targetNode) {
        return ''
      }

      // 获取节点连接点
      const sourcePoint = this.getNodeConnectionPoint(sourceNode, targetNode)
      const targetPoint = this.getNodeConnectionPoint(targetNode, sourceNode)

      // 确保点在画布范围内
      const sourceInBounds = this.clampPointToBounds(sourcePoint)
      const targetInBounds = this.clampPointToBounds(targetPoint)

      // 根据边样式计算路径
      switch (this.edgeStyle) {
        case 'straight':
          return `M ${sourceInBounds.x} ${sourceInBounds.y} L ${targetInBounds.x} ${targetInBounds.y}`

        case 'step':
          return this.calculateStepPath(sourceInBounds, targetInBounds)

        case 'dashed':
          return `M ${sourceInBounds.x} ${sourceInBounds.y} L ${targetInBounds.x} ${targetInBounds.y}`

        case 'smooth':
        default:
          return this.calculateSmoothPath(sourceInBounds, targetInBounds)
      }
    },

    /**
     * 限制点在画布边界内
     */
    clampPointToBounds(point) {
      return {
        x: Math.max(this.canvasBounds.minX + 20, Math.min(point.x, this.canvasBounds.maxX - 20)),
        y: Math.max(this.canvasBounds.minY + 20, Math.min(point.y, this.canvasBounds.maxY - 20))
      }
    },

    /**
     * 获取节点连接点
     */
    getNodeConnectionPoint(node, targetNode) {
      if (!targetNode) {
        return {
          x: node.position.x + (node.width || this.nodeWidth) / 2,
          y: node.position.y + (node.height || this.nodeHeight) / 2
        }
      }

      const nodeCenter = {
        x: node.position.x + (node.width || this.nodeWidth) / 2,
        y: node.position.y + (node.height || this.nodeHeight) / 2
      }

      const targetCenter = {
        x: targetNode.position.x + (targetNode.width || this.nodeWidth) / 2,
        y: targetNode.position.y + (targetNode.height || this.nodeHeight) / 2
      }

      // 计算方向向量
      const dx = targetCenter.x - nodeCenter.x
      const dy = targetCenter.y - nodeCenter.y

      // 确定从哪条边连接
      if (Math.abs(dx) > Math.abs(dy)) {
        // 水平方向
        if (dx > 0) {
          // 从右边连接
          return {
            x: node.position.x + (node.width || this.nodeWidth),
            y: nodeCenter.y
          }
        } else {
          // 从左边连接
          return {
            x: node.position.x,
            y: nodeCenter.y
          }
        }
      } else {
        // 垂直方向
        if (dy > 0) {
          // 从下边连接
          return {
            x: nodeCenter.x,
            y: node.position.y + (node.height || this.nodeHeight)
          }
        } else {
          // 从上边连接
          return {
            x: nodeCenter.x,
            y: node.position.y
          }
        }
      }
    },

    /**
     * 计算平滑路径
     */
    calculateSmoothPath(source, target) {
      const dx = target.x - source.x
      const dy = target.y - source.y
      const distance = Math.sqrt(dx * dx + dy * dy)

      // 根据距离调整控制点偏移
      const offset = Math.min(distance * 0.3, 100)

      if (Math.abs(dx) > Math.abs(dy)) {
        const cp1x = source.x + (dx > 0 ? offset : -offset)
        const cp1y = source.y
        const cp2x = target.x + (dx > 0 ? -offset : offset)
        const cp2y = target.y
        return `M ${source.x} ${source.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${target.x} ${target.y}`
      } else {
        const cp1x = source.x
        const cp1y = source.y + (dy > 0 ? offset : -offset)
        const cp2x = target.x
        const cp2y = target.y + (dy > 0 ? -offset : offset)
        return `M ${source.x} ${source.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${target.x} ${target.y}`
      }
    },

    /**
     * 计算阶梯路径
     */
    calculateStepPath(source, target) {
      const midX = (source.x + target.x) / 2
      const midY = (source.y + target.y) / 2

      return `M ${source.x} ${source.y} L ${midX} ${source.y} L ${midX} ${target.y} L ${target.x} ${target.y}`
    },

    /**
     * 获取边标签位置
     */
    getEdgeLabelPosition(edge) {
      const sourceNode = this.layoutedNodes.find(n => n.id === edge.source)
      const targetNode = this.layoutedNodes.find(n => n.id === edge.target)

      if (!sourceNode || !targetNode) return { x: 0, y: 0 }

      const sourceCenter = {
        x: sourceNode.position.x + (sourceNode.width || this.nodeWidth) / 2,
        y: sourceNode.position.y + (sourceNode.height || this.nodeHeight) / 2
      }

      const targetCenter = {
        x: targetNode.position.x + (targetNode.width || this.nodeWidth) / 2,
        y: targetNode.position.y + (targetNode.height || this.nodeHeight) / 2
      }

      // 计算连线的中间点
      const midX = (sourceCenter.x + targetCenter.x) / 2
      const midY = (sourceCenter.y + targetCenter.y) / 2

      // 确保标签在画布范围内
      return this.clampPointToBounds({ x: midX, y: midY })
    },

    /**
     * 获取边颜色
     */
    getEdgeColor(edge) {
      if (this.highlightedEdgeIds.has(edge.id)) return '#F56C6C'
      if (this.selectedEdgeId === edge.id) return '#F56C6C'
      
      // 根据边标签类型设置颜色
      const label = (edge.label || '').toLowerCase()
      if (label.includes('leads to') || label.includes('progresses')) {
        return '#409EFF'
      } else if (label.includes('uses') || label.includes('deploys')) {
        return '#67C23A'
      } else if (label.includes('targets') || label.includes('affects')) {
        return '#E6A23C'
      } else if (label.includes('exploits') || label.includes('creates')) {
        return '#F56C6C'
      } else if (label.includes('communicates') || label.includes('connects')) {
        return '#06b6d4'
      }
      
      return this.edgeColor
    },

    /**
     * 获取边宽度
     */
    getEdgeWidth(edge) {
      if (this.highlightedEdgeIds.has(edge.id)) return 3
      if (this.selectedEdgeId === edge.id) return 3
      
      // 根据边重要性设置宽度
      const label = (edge.label || '').toLowerCase()
      if (label.includes('leads to') || label.includes('uses')) {
        return 2.5
      }
      return 2
    },

    /**
     * 获取边虚线样式
     */
    getEdgeDashArray(edge) {
      if (this.edgeStyle === 'dashed') return '5,5'
      return ''
    },

    /**
     * 获取边箭头标记
     */
    getEdgeMarker(edge) {
      return 'url(#arrowhead-default)'
    },

    /**
     * 处理节点点击
     */
    handleNodeClick(node) {
      this.selectedNodeId = node.id
      this.selectedEdgeId = null
      this.$emit('node-click', node)
    },

    /**
     * 处理节点鼠标悬停
     */
    handleNodeMouseEnter(node) {
      this.highlightedNodeIds.add(node.id)

      // 高亮相关边
      this.edges.forEach(edge => {
        if (edge.source === node.id || edge.target === node.id) {
          this.highlightedEdgeIds.add(edge.id)
        }
      })
    },

    handleNodeMouseLeave(node) {
      this.highlightedNodeIds.delete(node.id)
      this.highlightedEdgeIds.clear()
    },

    /**
     * 处理边点击
     */
    handleEdgeClick(edge) {
      this.selectedEdgeId = edge.id
      this.selectedNodeId = null
      this.$emit('edge-click', edge)
    },

    /**
     * 处理边鼠标悬停
     */
    handleEdgeMouseEnter(edge) {
      this.highlightedEdgeIds.add(edge.id)
      this.highlightedNodeIds.add(edge.source)
      this.highlightedNodeIds.add(edge.target)
    },

    handleEdgeMouseLeave(edge) {
      this.highlightedEdgeIds.delete(edge.id)
      this.highlightedNodeIds.clear()
    },

    /**
     * 处理节点拖拽开始
     */
    handleNodeMouseDown(node, event) {
      event.stopPropagation()
      this.startDraggingNode(node, event)
    },

    /**
     * 处理连接点鼠标按下
     */
    handleHandleMouseDown(node, position) {
      this.drawingEdge = {
        source: node.id,
        sourcePosition: position,
        startX: node.position.x + (position === 'left' ? 0 : position === 'right' ? (node.width || this.nodeWidth) : (node.width || this.nodeWidth) / 2),
        startY: node.position.y + (position === 'top' ? 0 : position === 'bottom' ? (node.height || this.nodeHeight) : (node.height || this.nodeHeight) / 2)
      }
    },

    /**
     * 开始拖拽节点
     */
    startDraggingNode(node, event) {
      this.draggingNode = {
        node,
        offsetX: event.clientX - node.position.x,
        offsetY: event.clientY - node.position.y
      }

      event.currentTarget.style.cursor = 'grabbing'
    },

    /**
     * 处理画布鼠标按下
     */
    handleCanvasMouseDown(event) {
      if (event.target.closest('.flow-node')) return

      this.isPanning = true
      this.panStart = {
        x: event.clientX - this.pan.x,
        y: event.clientY - this.pan.y
      }

      event.currentTarget.style.cursor = 'grabbing'
      event.preventDefault()
    },

    /**
     * 处理画布鼠标进入
     */
    handleCanvasMouseEnter() {
      if (this.layoutedNodes.length > 10) {
        this.showScrollHint = true
        setTimeout(() => {
          this.showScrollHint = false
        }, 3000)
      }
    },

    handleCanvasMouseLeave() {
      this.showScrollHint = false
    },

    /**
     * 处理鼠标移动
     */
    handleMouseMove(event) {
      if (this.drawingEdge) {
        this.drawingEdgePath = `M ${this.drawingEdge.startX} ${this.drawingEdge.startY} L ${event.clientX} ${event.clientY}`
      }

      if (this.draggingNode) {
        const { node, offsetX, offsetY } = this.draggingNode
        const newX = event.clientX - offsetX
        const newY = event.clientY - offsetY

        // 更新节点位置
        const nodeIndex = this.layoutedNodes.findIndex(n => n.id === node.id)
        if (nodeIndex !== -1) {
          this.layoutedNodes[nodeIndex].position = { x: newX, y: newY }
          this.layoutedNodes = [...this.layoutedNodes]
          this.updateCanvasBounds()
        }
      }

      if (this.isPanning) {
        this.pan = {
          x: event.clientX - this.panStart.x,
          y: event.clientY - this.panStart.y
        }
      }
    },

    /**
     * 处理鼠标释放
     */
    handleMouseUp() {
      if (this.isPanning) {
        this.$refs.flowCanvas.style.cursor = 'grab'
      }
      if (this.draggingNode) {
        document.querySelectorAll('.flow-node').forEach(node => {
          node.style.cursor = 'pointer'
        })
      }

      this.isPanning = false
      this.draggingNode = null
      this.drawingEdge = null
      this.drawingEdgePath = ''
    },

    /**
     * 处理键盘事件
     */
    handleKeyDown(event) {
      if (event.key === 'Escape') {
        this.selectedNodeId = null
        this.selectedEdgeId = null
      } else if (event.key === 'Delete' && this.selectedNodeId) {
        this.deleteSelectedNode()
      }
    },

    /**
     * 删除选中节点
     */
    deleteSelectedNode() {
      if (!this.selectedNodeId) return

      this.layoutedNodes = this.layoutedNodes.filter(n => n.id !== this.selectedNodeId)

      const filteredEdges = this.edges.filter(e =>
        e.source !== this.selectedNodeId && e.target !== this.selectedNodeId
      )

      this.selectedNodeId = null
      this.updateCanvasBounds()
      this.$emit('nodes-updated', this.layoutedNodes)
      this.$emit('edges-updated', filteredEdges)
    },

    /**
     * 处理滚轮缩放
     */
    handleWheel(event) {
      event.preventDefault()

      const zoomSpeed = 0.1
      const delta = event.deltaY > 0 ? -zoomSpeed : zoomSpeed
      const newZoom = Math.max(0.05, Math.min(5, this.zoom + delta))

      if (this.$refs.flowWrapper) {
        const rect = this.$refs.flowWrapper.getBoundingClientRect()
        const mouseX = event.clientX - rect.left
        const mouseY = event.clientY - rect.top

        const scaleRatio = newZoom / this.zoom
        this.pan = {
          x: mouseX - (mouseX - this.pan.x) * scaleRatio,
          y: mouseY - (mouseY - this.pan.y) * scaleRatio
        }
      }

      this.zoom = newZoom
    },

    /**
     * 放大
     */
    zoomIn() {
      this.zoom = Math.min(5, this.zoom + 0.2)
    },

    /**
     * 缩小
     */
    zoomOut() {
      this.zoom = Math.max(0.05, this.zoom - 0.2)
    },

    /**
     * 适应视图
     */
    fitView() {
      if (this.layoutedNodes.length === 0 || !this.$refs.flowWrapper) return

      const container = this.$refs.flowWrapper
      const padding = 100
      
      const contentWidth = this.canvasBounds.width + padding * 2
      const contentHeight = this.canvasBounds.height + padding * 2
      const containerWidth = container.clientWidth
      const containerHeight = container.clientHeight

      const scaleX = containerWidth / contentWidth
      const scaleY = containerHeight / contentHeight
      const scale = Math.min(scaleX, scaleY, 1) * 0.85

      this.zoom = Math.max(0.1, Math.min(3, scale))

      const scaledWidth = contentWidth * this.zoom
      const scaledHeight = contentHeight * this.zoom

      this.pan = {
        x: (containerWidth - scaledWidth) / 2 - (this.canvasBounds.minX - padding) * this.zoom,
        y: (containerHeight - scaledHeight) / 2 - (this.canvasBounds.minY - padding) * this.zoom
      }
    },

    /**
     * 重置视图
     */
    resetView() {
      this.zoom = 1.0
      this.pan = { x: 0, y: 0 }
    },

    /**
     * 切换网格显示
     */
    toggleGrid() {
      this.showGrid = !this.showGrid
    },

    /**
     * 切换标签显示
     */
    toggleLabels() {
      this.showLabels = !this.showLabels
    },

    /**
     * 切换边标签显示
     */
    toggleEdgeLabels() {
      this.showEdgeLabels = !this.showEdgeLabels
    },

    /**
     * 切换节点详情显示
     */
    toggleNodeDetails() {
      this.showNodeDetails = !this.showNodeDetails
    },

    /**
     * 切换收藏
     */
    toggleFavorite(node) {
      const nodeIndex = this.layoutedNodes.findIndex(n => n.id === node.id)
      if (nodeIndex !== -1) {
        this.layoutedNodes[nodeIndex].favorite = !this.layoutedNodes[nodeIndex].favorite
        this.layoutedNodes = [...this.layoutedNodes]
      }
    },

    /**
     * 导出图表
     */
    async exportDiagram(format = 'png') {
      try {
        if (format === 'png') {
          await this.exportAsPNG()
        } else if (format === 'json') {
          this.exportAsJSON()
        } else if (format === 'attackflow') {
          this.exportAsAttackFlow()
        } else if (format === 'stix') {
          this.exportAsSTIX()
        }

        this.$emit('exported', { format, success: true })
      } catch (error) {
        console.error('导出失败:', error)
        this.$emit('exported', { format, success: false, error })
        throw error
      }
    },

    /**
     * 导出为PNG
     */
    async exportAsPNG() {
      const html2canvas = await import('html2canvas').catch(() => {
        throw new Error('请先安装html2canvas: npm install html2canvas')
      })

      // 临时保存当前状态
      const originalZoom = this.zoom
      const originalPan = { ...this.pan }
      
      // 先调整到适合导出的视图
      this.fitView()
      
      await this.$nextTick()
      
      const element = this.$refs.flowCanvas
      if (!element) {
        throw new Error('找不到画布元素')
      }

      try {
        // 创建临时容器用于导出
        const tempContainer = document.createElement('div')
        tempContainer.style.width = `${this.canvasBounds.width}px`
        tempContainer.style.height = `${this.canvasBounds.height}px`
        tempContainer.style.background = 'white'
        tempContainer.style.overflow = 'hidden'
        tempContainer.style.position = 'relative'
        
        // 克隆画布内容
        const clonedCanvas = element.cloneNode(true)
        clonedCanvas.style.transform = 'none'
        clonedCanvas.style.position = 'absolute'
        clonedCanvas.style.top = '0'
        clonedCanvas.style.left = '0'
        clonedCanvas.style.width = `${this.canvasBounds.width}px`
        clonedCanvas.style.height = `${this.canvasBounds.height}px`
        
        tempContainer.appendChild(clonedCanvas)
        document.body.appendChild(tempContainer)

        const canvas = await html2canvas.default(tempContainer, {
          backgroundColor: '#ffffff',
          scale: 2,
          useCORS: true,
          logging: false,
          width: this.canvasBounds.width,
          height: this.canvasBounds.height
        })

        const dataUrl = canvas.toDataURL('image/png')
        const link = document.createElement('a')
        link.download = `attack-flow-${Date.now()}.png`
        link.href = dataUrl
        link.click()

        document.body.removeChild(tempContainer)
        
        this.$message.success('PNG导出成功')
      } finally {
        // 恢复原始视图
        this.zoom = originalZoom
        this.pan = originalPan
      }
    },

    /**
     * 导出为JSON
     */
    exportAsJSON() {
      const data = {
        nodes: this.layoutedNodes.map(node => ({
          id: node.id,
          type: node.type,
          data: node.data,
          position: node.position,
          width: node.width,
          height: node.height
        })),
        edges: this.edges.map(edge => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          relation: edge.relation,
          data: edge.data
        })),
        metadata: {
          exportedAt: new Date().toISOString(),
          tool: 'FlowViz Professional',
          version: '1.0.0',
          nodeCount: this.layoutedNodes.length,
          edgeCount: this.edges.length,
          layoutDirection: this.layoutDirection,
          canvasBounds: this.canvasBounds,
          format: 'flowviz_extended'
        }
      }

      const dataStr = JSON.stringify(data, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)

      const link = document.createElement('a')
      link.download = `attack-flow-${Date.now()}.json`
      link.href = url
      link.click()

      URL.revokeObjectURL(url)

      this.$message.success('JSON导出成功')
    },

    /**
     * 导出为ATT&CK Flow格式
     */
    exportAsAttackFlow() {
      const attackFlow = {
        spec_version: '3.0',
        objects: [],
        data_components: [],
        attack_flow: {
          nodes: this.layoutedNodes.map(node => ({
            id: node.id,
            type: node.type,
            name: node.data.label || node.data.name || node.id,
            description: node.data.description || '',
            metadata: {
              technique_id: node.data.technique_id,
              tactic: node.data.tactic,
              confidence: node.data.confidence,
              risk_level: node.data.risk_level,
              position: node.position
            }
          })),
          edges: this.edges.map(edge => ({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.label || edge.relation || '关联',
            description: edge.data?.description || '',
            timestamp: edge.data?.timestamp || new Date().toISOString()
          })),
          layout: {
            direction: this.layoutDirection,
            bounds: this.canvasBounds
          }
        }
      }

      const dataStr = JSON.stringify(attackFlow, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)

      const link = document.createElement('a')
      link.download = `attack-flow-v3-${Date.now()}.json`
      link.href = url
      link.click()

      URL.revokeObjectURL(url)

      this.$message.success('ATT&CK Flow格式导出成功')
    },

    /**
     * 导出为STIX格式
     */
    exportAsSTIX() {
      const stixBundle = {
        type: 'bundle',
        id: `bundle--${this.generateUUID()}`,
        spec_version: '2.1',
        objects: [
          {
            type: 'identity',
            id: `identity--${this.generateUUID()}`,
            name: 'FlowViz Export',
            identity_class: 'organization'
          },
          ...this.layoutedNodes.map(node => ({
            type: 'indicator',
            id: `indicator--${this.generateUUID()}`,
            name: node.data.label || node.data.name || node.id,
            description: node.data.description || '',
            pattern: this.getStixPatternForNode(node),
            pattern_type: 'stix',
            valid_from: new Date().toISOString(),
            labels: [node.type]
          }))
        ]
      }

      const dataStr = JSON.stringify(stixBundle, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)

      const link = document.createElement('a')
      link.download = `stix-bundle-${Date.now()}.json`
      link.href = url
      link.click()

      URL.revokeObjectURL(url)

      this.$message.success('STIX格式导出成功')
    },

    /**
     * 生成UUID
     */
    generateUUID() {
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0
        const v = c === 'x' ? r : (r & 0x3 | 0x8)
        return v.toString(16)
      })
    },

    /**
     * 获取节点的STIX模式
     */
    getStixPatternForNode(node) {
      const patterns = []

      if (node.data.ip_address) {
        patterns.push(`[ipv4-addr:value = '${node.data.ip_address}']`)
      }

      if (node.data.hash) {
        patterns.push(`[file:hashes.'SHA-256' = '${node.data.hash}']`)
      }

      if (node.data.url) {
        patterns.push(`[url:value = '${node.data.url}']`)
      }

      if (node.data.registry_key) {
        patterns.push(`[windows-registry-key:key = '${node.data.registry_key}']`)
      }

      return patterns.length > 0 ? patterns.join(' OR ') : `[threat-actor:name = '${node.data.label}']`
    }
  }
}
</script>

<style scoped>
.flow-visualization-wrapper {
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
  background: #ffffff;
  border-radius: 8px;
  border: 1px solid #ebeef5;
}

/* 工具栏切换按钮 */
.toolbar-toggle {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 1001;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* 工具栏 */
.flow-toolbar {
  position: absolute;
  top: 50px;
  left: 10px;
  z-index: 1000;
  background: white;
  border-radius: 8px;
  padding: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border: 1px solid #ebeef5;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.toolbar-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.toolbar-group:not(:last-child) {
  padding-bottom: 8px;
  border-bottom: 1px solid #ebeef5;
}

/* 滚动提示 */
.scroll-hint {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 100;
  background: rgba(255, 255, 255, 0.9);
  padding: 8px 16px;
  border-radius: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border: 1px solid #409EFF;
  display: flex;
  align-items: center;
  gap: 8px;
  animation: fadeInOut 3s ease-in-out;
  font-size: 12px;
  color: #409EFF;
}

@keyframes fadeInOut {
  0%, 100% { opacity: 0; }
  20%, 80% { opacity: 1; }
}

/* 缩放显示 */
.zoom-display {
  position: absolute;
  bottom: 20px;
  right: 20px;
  background: white;
  color: #606266;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid #ebeef5;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
  z-index: 100;
}

/* 空状态 */
.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  z-index: 10;
}

.empty-state-content {
  background: white;
  padding: 30px;
  border-radius: 12px;
  border: 1px solid #ebeef5;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.empty-icon {
  font-size: 48px;
  color: #c0c4cc;
  margin-bottom: 15px;
}

.empty-state h3 {
  margin-bottom: 10px;
  color: #303133;
  font-size: 18px;
}

.empty-state p {
  color: #606266;
  margin-bottom: 5px;
}

.hint {
  font-size: 12px;
  color: #909399;
  font-style: italic;
}

/* 主画布 - 修复：移除固定最小尺寸，使用动态尺寸 */
.flow-canvas {
  position: absolute;
  top: 0;
  left: 0;
  cursor: grab;
  user-select: none;
  transition: transform 0.15s ease;
  background: #ffffff;
  transform-origin: 0 0;
}

.flow-canvas:active {
  cursor: grabbing;
}

/* 网格背景 */
.grid-background {
  position: absolute;
  top: 0;
  left: 0;
  background-image:
    linear-gradient(rgba(0, 0, 0, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 0, 0, 0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}

/* 节点样式 */
.flow-node {
  position: absolute;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: all 0.3s ease;
  border: 2px solid #ebeef5;
  border-left-width: 4px;
  overflow: hidden;
  z-index: 10;
}

.flow-node:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.flow-node.node-selected {
  border-color: #F56C6C !important;
  box-shadow: 0 0 0 2px rgba(245, 108, 108, 0.1), 0 4px 16px rgba(0, 0, 0, 0.15);
}

.flow-node.node-highlighted {
  border-color: #F56C6C !important;
  box-shadow: 0 0 0 2px rgba(245, 108, 108, 0.2);
}

/* 节点连接点 */
.node-handle {
  position: absolute;
  width: 10px;
  height: 10px;
  background: white;
  border-radius: 50%;
  border: 2px solid #409EFF;
  z-index: 11;
  cursor: crosshair;
  opacity: 0;
  transition: all 0.2s ease;
}

.node-handle:hover {
  background: #409EFF;
  transform: scale(1.2);
  opacity: 1;
}

.flow-node:hover .node-handle {
  opacity: 1;
}

.node-handle-top {
  top: -5px;
  left: 50%;
  transform: translateX(-50%);
}

.node-handle-bottom {
  bottom: -5px;
  left: 50%;
  transform: translateX(-50%);
}

.node-handle-left {
  left: -5px;
  top: 50%;
  transform: translateY(-50%);
}

.node-handle-right {
  right: -5px;
  top: 50%;
  transform: translateY(-50%);
}

/* 节点头部 */
.node-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 12px 16px;
  color: white;
  min-height: 40px;
  border-radius: 6px 6px 0 0;
}

.node-title-wrapper {
  flex: 1;
  margin-right: 10px;
  overflow: hidden;
}

.node-title {
  font-size: 14px;
  font-weight: bold;
  line-height: 1.4;
  margin-bottom: 4px;
  color: white;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.node-type-badge {
  font-size: 10px;
  padding: 2px 8px;
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border-radius: 12px;
  text-transform: uppercase;
  display: inline-block;
}

.node-actions {
  cursor: pointer;
  padding: 2px;
}

.node-actions i {
  font-size: 16px;
  color: rgba(255, 255, 255, 0.8);
  transition: color 0.2s;
}

.node-actions i:hover {
  color: white;
}

/* MITRE信息 */
.mitre-info {
  margin-bottom: 10px;
}

.mitre-technique {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.technique-tag {
  font-weight: bold;
  border: none;
}

.tactic-text {
  font-size: 11px;
  color: #606266;
  font-style: italic;
}

/* 节点内容 */
.node-content {
  padding: 12px 16px;
  background: white;
}

.node-description {
  font-size: 12px;
  color: #606266;
  line-height: 1.5;
  margin-bottom: 12px;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.node-technical {
  margin-bottom: 10px;
}

.technical-item {
  display: flex;
  align-items: center;
  margin-bottom: 6px;
}

.technical-label {
  font-size: 11px;
  color: #909399;
  margin-right: 8px;
  min-width: 35px;
}

.technical-value {
  flex: 1;
  font-size: 11px;
}

.technical-value.code {
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  background: #f5f5f5;
  padding: 2px 6px;
  border-radius: 4px;
  word-break: break-all;
}

.node-confidence {
  display: flex;
  justify-content: flex-start;
}

.confidence-tag {
  font-size: 10px !important;
  height: 18px !important;
  line-height: 18px !important;
}

/* 流式处理指示器 */
.node-streaming-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  font-size: 10px;
  color: #409EFF;
  margin-top: 8px;
  padding: 4px;
  background: #f0f7ff;
  border-radius: 4px;
  animation: pulse 2s infinite;
}

.node-streaming-indicator i {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 节点底部 */
.node-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  background: #f8f9fa;
  border-top: 1px solid #ebeef5;
  font-size: 11px;
  color: #909399;
}

.source-indicator,
.connections-count {
  display: flex;
  align-items: center;
  gap: 4px;
}

/* SVG容器 */
.edges-container,
.highlight-edges-container {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
  z-index: 5;
}

.highlight-edges-container {
  z-index: 4;
}

/* 连线样式 */
.flow-edge {
  transition: all 0.3s ease;
  pointer-events: all;
  cursor: pointer;
}

.flow-edge:hover {
  stroke-width: 3 !important;
  filter: drop-shadow(0 0 2px currentColor);
}

.edge-highlighted {
  stroke: #F56C6C !important;
  stroke-width: 3 !important;
}

.edge-selected {
  stroke: #F56C6C !important;
  stroke-width: 3 !important;
  filter: drop-shadow(0 0 3px rgba(245, 108, 108, 0.5));
}

.edge-animated {
  stroke-dasharray: 10, 5;
  animation: dash 20s linear infinite;
}

@keyframes dash {
  to {
    stroke-dashoffset: 1000;
  }
}

.highlight-edge {
  pointer-events: none;
}

/* 连线标签 */
.edge-label-container {
  pointer-events: none;
}

.edge-label-bg {
  pointer-events: none;
  opacity: 0.95;
  filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.15));
}

.edge-label {
  pointer-events: none;
  user-select: none;
  font-weight: 600;
  text-shadow: 0 0 2px white;
}

/* 正在绘制的连线 */
.drawing-edge {
  pointer-events: none;
  animation: dash-animation 1s linear infinite;
}

@keyframes dash-animation {
  from {
    stroke-dashoffset: 0;
  }
  to {
    stroke-dashoffset: 10;
  }
}

/* 流式处理指示器 */
.streaming-indicator {
  position: absolute;
  top: 20px;
  right: 20px;
  display: flex;
  align-items: center;
  background: white;
  padding: 8px 16px;
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

.node-count,
.edge-count {
  font-size: 12px;
  color: #909399;
  background: #f5f7fa;
  padding: 2px 6px;
  border-radius: 10px;
  margin-left: 5px;
}

/* 统计面板 */
.stats-panel {
  position: absolute;
  bottom: 20px;
  left: 20px;
  background: white;
  padding: 8px 12px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border: 1px solid #ebeef5;
  display: flex;
  gap: 15px;
  z-index: 100;
  flex-wrap: wrap;
}

.stats-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stats-label {
  font-size: 10px;
  color: #909399;
  margin-bottom: 2px;
}

.stats-value {
  font-size: 14px;
  font-weight: bold;
  color: #409EFF;
}

/* 节点类型颜色样式 */
.node-type-action {
  border-left-color: #409EFF !important;
}

.node-type-tool {
  border-left-color: #67C23A !important;
}

.node-type-malware {
  border-left-color: #F56C6C !important;
}

.node-type-asset {
  border-left-color: #E6A23C !important;
}

.node-type-infrastructure {
  border-left-color: #06b6d4 !important;
}

.node-type-vulnerability {
  border-left-color: #f43f5e !important;
}

.node-type-file {
  border-left-color: #8b5cf6 !important;
}

.node-type-process {
  border-left-color: #10b981 !important;
}

.node-type-network {
  border-left-color: #3b82f6 !important;
}

.node-type-registry {
  border-left-color: #8b5cf6 !important;
}

.node-type-AND_operator,
.node-type-OR_operator {
  border-left-color: #909399 !important;
  min-width: 100px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .flow-toolbar {
    left: 5px;
    top: 5px;
    padding: 6px;
  }

  .toolbar-group {
    flex-wrap: wrap;
  }

  .zoom-display {
    bottom: 10px;
    right: 10px;
  }

  .stats-panel {
    bottom: 10px;
    left: 10px;
    padding: 6px 10px;
    gap: 10px;
  }

  .streaming-indicator {
    top: 10px;
    right: 10px;
    padding: 6px 12px;
    flex-wrap: wrap;
  }
  
  .scroll-hint {
    display: none;
  }
}

/* 小屏幕优化 */
@media (max-width: 480px) {
  .node-header {
    padding: 8px 12px;
  }
  
  .node-content {
    padding: 8px 12px;
  }
  
  .node-footer {
    padding: 6px 12px;
  }
  
  .stats-panel {
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .stats-item {
    min-width: 60px;
  }
}
</style>