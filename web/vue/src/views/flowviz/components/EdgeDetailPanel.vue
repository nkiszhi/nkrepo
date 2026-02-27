<!-- vue/src/views/flowviz/components/EdgeDetailPanel.vue -->
<template>
  <div class="edge-detail-panel">
    <div v-if="edge" class="detail-content">
      <!-- 基本信息 -->
      <div class="section">
        <h3 class="section-title">连接信息</h3>
        <div class="info-grid">
          <div class="info-item">
            <span class="label">连接ID</span>
            <span class="value code">{{ edge.id }}</span>
          </div>
          <div class="info-item">
            <span class="label">关系类型</span>
            <el-tag size="small" :type="getEdgeTypeTag(edge.label)">
              {{ edge.label || '关联' }}
            </el-tag>
          </div>
          <div v-if="edge.type" class="info-item">
            <span class="label">连线样式</span>
            <el-tag size="small" type="info">
              {{ getEdgeStyleLabel(edge.type) }}
            </el-tag>
          </div>
        </div>
      </div>

      <!-- 连接节点 -->
      <div class="section">
        <h3 class="section-title">连接节点</h3>
        <div class="nodes-grid">
          <div class="node-info source-node">
            <div class="node-header">
              <span class="node-label">源节点</span>
            </div>
            <div v-if="sourceNode" class="node-content">
              <div class="node-name">{{ sourceNode.data.label || sourceNode.data.name || sourceNode.id }}</div>
              <div class="node-type">
                <el-tag size="small" :type="getNodeTypeTag(sourceNode.type)">
                  {{ getNodeTypeLabel(sourceNode.type) }}
                </el-tag>
              </div>
              <div v-if="sourceNode.data.description" class="node-desc">
                {{ truncateText(sourceNode.data.description, 100) }}
              </div>
              <div v-if="sourceNode.data.technique_id" class="node-technique">
                <el-tag size="small" type="danger">
                  {{ sourceNode.data.technique_id }}
                </el-tag>
              </div>
            </div>
            <div v-else class="node-missing">
              节点不存在
            </div>
          </div>

          <div class="connection-arrow">
            <i class="el-icon-right" />
            <div class="edge-direction">→</div>
          </div>

          <div class="node-info target-node">
            <div class="node-header">
              <span class="node-label">目标节点</span>
            </div>
            <div v-if="targetNode" class="node-content">
              <div class="node-name">{{ targetNode.data.label || targetNode.data.name || targetNode.id }}</div>
              <div class="node-type">
                <el-tag size="small" :type="getNodeTypeTag(targetNode.type)">
                  {{ getNodeTypeLabel(targetNode.type) }}
                </el-tag>
              </div>
              <div v-if="targetNode.data.description" class="node-desc">
                {{ truncateText(targetNode.data.description, 100) }}
              </div>
              <div v-if="targetNode.data.technique_id" class="node-technique">
                <el-tag size="small" type="danger">
                  {{ targetNode.data.technique_id }}
                </el-tag>
              </div>
            </div>
            <div v-else class="node-missing">
              节点不存在
            </div>
          </div>
        </div>
      </div>

      <!-- 攻击关系详情 -->
      <div v-if="edge.data" class="section">
        <h3 class="section-title">关系详情</h3>
        <div class="relationship-info">
          <div v-if="edge.data.description" class="description">
            {{ edge.data.description }}
          </div>
          
          <div class="relationship-meta">
            <div v-if="edge.data.timestamp" class="meta-item">
              <span class="label">时间戳:</span>
              <span class="value">{{ formatTimestamp(edge.data.timestamp) }}</span>
            </div>
            
            <div v-if="edge.data.evidence" class="meta-item">
              <span class="label">证据:</span>
              <span class="value">{{ edge.data.evidence }}</span>
            </div>
            
            <div v-if="edge.data.confidence" class="meta-item">
              <span class="label">置信度:</span>
              <el-tag size="small" :type="getConfidenceTag(edge.data.confidence)">
                {{ getConfidenceLabel(edge.data.confidence) }}
              </el-tag>
            </div>
            
            <div v-if="edge.data.order" class="meta-item">
              <span class="label">顺序:</span>
              <span class="value">{{ edge.data.order }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- MITRE ATT&CK信息 -->
      <div v-if="hasMitreInfo" class="section">
        <h3 class="section-title">MITRE ATT&CK 上下文</h3>
        <div class="mitre-info">
          <div v-if="sourceNode && sourceNode.type === 'action'" class="mitre-item">
            <h4>源节点技术:</h4>
            <div class="technique-info">
              <el-tag size="small" type="danger">{{ sourceNode.data.technique_id }}</el-tag>
              <span class="tactic">{{ sourceNode.data.tactic }}</span>
            </div>
          </div>
          
          <div v-if="targetNode && targetNode.type === 'action'" class="mitre-item">
            <h4>目标节点技术:</h4>
            <div class="technique-info">
              <el-tag size="small" type="danger">{{ targetNode.data.technique_id }}</el-tag>
              <span class="tactic">{{ targetNode.data.tactic }}</span>
            </div>
          </div>
          
          <div v-if="edge.data && edge.data.attack_flow" class="mitre-item">
            <h4>攻击流程:</h4>
            <div class="flow-description">{{ edge.data.attack_flow }}</div>
          </div>
        </div>
      </div>

      <!-- 操作 -->
      <div class="section">
        <h3 class="section-title">操作</h3>
        <div class="actions">
          <el-button size="small" type="primary" @click="focusOnEdge">
            <i class="el-icon-view" style="margin-right: 5px;" />
            聚焦查看
          </el-button>
          <el-button size="small" @click="exportEdge">
            <i class="el-icon-download" style="margin-right: 5px;" />
            导出信息
          </el-button>
          <el-button size="small" @click="copyEdgeInfo">
            <i class="el-icon-document-copy" style="margin-right: 5px;" />
            复制信息
          </el-button>
        </div>
      </div>
    </div>
    
    <div v-else class="no-edge-selected">
      <el-empty description="未选择连接" :image-size="100" />
    </div>
  </div>
</template>

<script>
export default {
  name: 'EdgeDetailPanel',
  props: {
    edge: {
      type: Object,
      default: null
    },
    nodes: {
      type: Array,
      default: () => []
    }
  },

  computed: {
    sourceNode() {
      return this.nodes.find(node => node.id === this.edge?.source)
    },

    targetNode() {
      return this.nodes.find(node => node.id === this.edge?.target)
    },
    
    hasMitreInfo() {
      return (this.sourceNode && this.sourceNode.type === 'action') || 
             (this.targetNode && this.targetNode.type === 'action')
    }
  },

  methods: {
    getNodeTypeTag(type) {
      const map = {
        'action': 'primary',
        'tool': 'success',
        'malware': 'danger',
        'asset': 'warning',
        'infrastructure': 'info',
        'vulnerability': 'danger',
        'file': '',
        'process': 'success',
        'network': 'info',
        'registry': '',
        'AND_operator': 'info',
        'OR_operator': 'info',
        'default': 'info'
      }
      return map[type] || 'info'
    },

    getNodeTypeLabel(type) {
      const map = {
        'action': '攻击行动',
        'tool': '工具',
        'malware': '恶意软件',
        'asset': '资产',
        'infrastructure': '基础设施',
        'vulnerability': '漏洞',
        'file': '文件',
        'process': '进程',
        'network': '网络',
        'registry': '注册表',
        'AND_operator': 'AND门',
        'OR_operator': 'OR门',
        'default': '节点'
      }
      return map[type] || type
    },
    
    getEdgeTypeTag(label) {
      const labelLower = (label || '').toLowerCase()
      if (labelLower.includes('uses') || labelLower.includes('deploys')) {
        return 'success'
      } else if (labelLower.includes('targets') || labelLower.includes('affects')) {
        return 'warning'
      } else if (labelLower.includes('leads to') || labelLower.includes('progresses')) {
        return 'primary'
      } else if (labelLower.includes('exploits') || labelLower.includes('creates')) {
        return 'danger'
      } else if (labelLower.includes('communicates') || labelLower.includes('connects')) {
        return 'info'
      }
      return ''
    },
    
    getEdgeStyleLabel(type) {
      const map = {
        'floating': '浮动线',
        'straight': '直线',
        'step': '阶梯线',
        'smoothstep': '平滑线'
      }
      return map[type] || type
    },
    
    getConfidenceTag(confidence) {
      const map = {
        'high': 'success',
        'medium': 'warning',
        'low': 'danger'
      }
      return map[confidence] || 'info'
    },
    
    getConfidenceLabel(confidence) {
      const map = {
        'high': '高置信度',
        'medium': '中置信度',
        'low': '低置信度'
      }
      return map[confidence] || confidence
    },

    formatTimestamp(timestamp) {
      if (!timestamp) return ''
      try {
        const date = new Date(timestamp)
        return date.toLocaleString()
      } catch {
        return timestamp
      }
    },
    
    truncateText(text, maxLength) {
      if (!text) return ''
      if (text.length <= maxLength) return text
      return text.substring(0, maxLength) + '...'
    },

    focusOnEdge() {
      this.$emit('focus-edge', this.edge)
    },

    exportEdge() {
      const edgeData = {
        edge: this.edge,
        sourceNode: this.sourceNode,
        targetNode: this.targetNode,
        exportedAt: new Date().toISOString(),
        format: 'FlowViz Edge Export'
      }

      const dataStr = JSON.stringify(edgeData, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)

      const link = document.createElement('a')
      link.download = `edge-${this.edge.id}-${Date.now()}.json`
      link.href = url
      link.click()

      URL.revokeObjectURL(url)

      this.$message.success('连接信息已导出')
    },
    
    copyEdgeInfo() {
      const edgeInfo = {
        '连接ID': this.edge.id,
        '关系类型': this.edge.label,
        '源节点': this.sourceNode ? (this.sourceNode.data.label || this.sourceNode.id) : '未知',
        '目标节点': this.targetNode ? (this.targetNode.data.label || this.targetNode.id) : '未知',
        '描述': this.edge.data?.description || '无'
      }
      
      const text = Object.entries(edgeInfo)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\n')
      
      navigator.clipboard.writeText(text)
        .then(() => {
          this.$message.success('连接信息已复制到剪贴板')
        })
        .catch(err => {
          console.error('复制失败:', err)
          this.$message.error('复制失败')
        })
    }
  }
}
</script>

<style scoped>
.edge-detail-panel {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
  background: white;
}

.detail-content {
  max-width: 800px;
  margin: 0 auto;
}

.section {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #ebeef5;
}

.section:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.section-title {
  color: #303133;
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
}

.section-title::before {
  content: '';
  display: inline-block;
  width: 4px;
  height: 16px;
  background: #409EFF;
  border-radius: 2px;
  margin-right: 8px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.label {
  font-size: 12px;
  color: #909399;
  font-weight: 500;
}

.value {
  font-size: 14px;
  color: #303133;
  font-weight: 500;
}

.code {
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  background: #f5f5f5;
  padding: 4px 8px;
  border-radius: 4px;
  word-break: break-all;
}

/* 节点网格 */
.nodes-grid {
  display: flex;
  align-items: center;
  gap: 20px;
}

.node-info {
  flex: 1;
  background: white;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #ebeef5;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.node-header {
  background: #f5f7fa;
  padding: 8px 12px;
  border-bottom: 1px solid #ebeef5;
}

.node-label {
  font-size: 12px;
  color: #909399;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.node-content {
  padding: 12px;
}

.node-name {
  font-size: 14px;
  color: #303133;
  font-weight: 600;
  margin-bottom: 8px;
}

.node-type {
  margin-bottom: 8px;
}

.node-desc {
  font-size: 12px;
  color: #606266;
  line-height: 1.4;
  margin-bottom: 8px;
  opacity: 0.8;
}

.node-technique {
  margin-top: 8px;
}

.node-missing {
  padding: 20px;
  text-align: center;
  color: #909399;
  font-style: italic;
}

.connection-arrow {
  color: #409EFF;
  font-size: 24px;
  opacity: 0.7;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
}

.edge-direction {
  font-size: 18px;
  font-weight: bold;
  color: #F56C6C;
}

/* 关系信息 */
.relationship-info {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.description {
  font-size: 14px;
  color: #606266;
  line-height: 1.6;
  background: #f8f9fa;
  padding: 12px;
  border-radius: 8px;
  border-left: 3px solid #409EFF;
}

.relationship-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  background: #f8f9fa;
  border-radius: 6px;
}

/* MITRE信息 */
.mitre-info {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.mitre-item {
  padding: 12px;
  background: #f0f9ff;
  border-radius: 8px;
  border-left: 4px solid #409EFF;
}

.mitre-item h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #303133;
}

.technique-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.tactic {
  font-size: 13px;
  color: #606266;
  font-style: italic;
}

.flow-description {
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
}

/* 操作 */
.actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.no-edge-selected {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
}

/* 滚动条样式 */
.edge-detail-panel::-webkit-scrollbar {
  width: 6px;
}

.edge-detail-panel::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.edge-detail-panel::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.edge-detail-panel::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

@media (max-width: 768px) {
  .nodes-grid {
    flex-direction: column;
  }
  
  .connection-arrow {
    transform: rotate(90deg);
    margin: 10px 0;
  }
  
  .actions {
    flex-direction: column;
  }
  
  .actions > button {
    width: 100%;
  }
}
</style>