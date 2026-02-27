<!-- vue/src/views/flowviz/components/NodeDetailPanel.vue -->
<template>
  <div class="node-detail-panel">
    <div v-if="node" class="detail-content">
      <!-- 基本信息 -->
      <div class="section">
        <h3 class="section-title">基本信息</h3>
        <div class="info-grid">
          <div class="info-item">
            <span class="label">节点ID</span>
            <span class="value">{{ node.id }}</span>
          </div>
          <div class="info-item">
            <span class="label">节点类型</span>
            <el-tag :type="getNodeTypeTag(node.type)" size="small">
              {{ getNodeTypeLabel(node.type) }}
            </el-tag>
          </div>
          <div class="info-item">
            <span class="label">节点名称</span>
            <span class="value">{{ node.data.label || node.data.name || '未命名' }}</span>
          </div>
        </div>
      </div>

      <!-- 描述信息 -->
      <div v-if="node.data.description" class="section">
        <h3 class="section-title">描述</h3>
        <div class="description-text">
          {{ node.data.description }}
        </div>
      </div>

      <!-- MITRE ATT&CK 信息 -->
      <div v-if="node.data.technique_id || node.data.tactic" class="section">
        <h3 class="section-title">MITRE ATT&CK</h3>
        <div class="mitre-info">
          <div v-if="node.data.technique_id" class="info-item">
            <span class="label">技术ID</span>
            <el-tag type="danger" size="small">
              {{ node.data.technique_id }}
            </el-tag>
          </div>
          <div v-if="node.data.tactic" class="info-item">
            <span class="label">攻击战术</span>
            <el-tag type="warning" size="small">
              {{ node.data.tactic }}
            </el-tag>
          </div>
        </div>
      </div>

      <!-- 技术指标 -->
      <div v-if="hasTechnicalIndicators" class="section">
        <h3 class="section-title">技术指标</h3>
        <div class="indicators-grid">
          <div v-if="node.data.ip_address" class="indicator-item">
            <span class="label">IP地址</span>
            <span class="value code">{{ node.data.ip_address }}</span>
          </div>
          <div v-if="node.data.file_path" class="indicator-item">
            <span class="label">文件路径</span>
            <span class="value code">{{ node.data.file_path }}</span>
          </div>
          <div v-if="node.data.hash" class="indicator-item">
            <span class="label">文件哈希</span>
            <span class="value code">{{ formatHash(node.data.hash) }}</span>
          </div>
          <div v-if="node.data.url" class="indicator-item">
            <span class="label">URL</span>
            <span class="value code">{{ node.data.url }}</span>
          </div>
          <div v-if="node.data.registry_key" class="indicator-item">
            <span class="label">注册表键</span>
            <span class="value code">{{ node.data.registry_key }}</span>
          </div>
        </div>
      </div>

      <!-- 元数据 -->
      <div v-if="hasMetadata" class="section">
        <h3 class="section-title">元数据</h3>
        <div class="metadata-grid">
          <div v-if="node.data.confidence" class="metadata-item">
            <span class="label">置信度</span>
            <el-tag :type="getConfidenceType(node.data.confidence)" size="small">
              {{ getConfidenceText(node.data.confidence) }}
            </el-tag>
          </div>
          <div v-if="node.data.risk_level" class="metadata-item">
            <span class="label">风险等级</span>
            <el-tag :type="getRiskType(node.data.risk_level)" size="small">
              {{ node.data.risk_level }}
            </el-tag>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'NodeDetailPanel',
  props: {
    node: {
      type: Object,
      default: null
    }
  },

  computed: {
    hasTechnicalIndicators() {
      return (
        this.node?.data?.ip_address ||
        this.node?.data?.file_path ||
        this.node?.data?.hash ||
        this.node?.data?.url ||
        this.node?.data?.registry_key
      )
    },

    hasMetadata() {
      return (
        this.node?.data?.confidence ||
        this.node?.data?.risk_level
      )
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
        'AND_operator': '',
        'OR_operator': ''
      }
      return map[type] || 'info'
    },

    getNodeTypeLabel(type) {
      const map = {
        'action': '行动',
        'tool': '工具',
        'malware': '恶意软件',
        'asset': '资产',
        'infrastructure': '基础设施',
        'url': 'URL',
        'vulnerability': '漏洞',
        'AND_operator': 'AND 门',
        'OR_operator': 'OR 门',
        'default': '节点'
      }
      return map[type] || type
    },

    formatHash(hash) {
      if (!hash) return ''
      if (hash.length > 32) {
        return hash.substring(0, 16) + '...' + hash.substring(hash.length - 8)
      }
      return hash
    },

    getConfidenceType(confidence) {
      const map = {
        'high': 'success',
        'medium': 'warning',
        'low': 'danger',
        'info': 'info'
      }
      return map[confidence] || 'info'
    },

    getConfidenceText(confidence) {
      const map = {
        'high': '高置信度',
        'medium': '中置信度',
        'low': '低置信度',
        'info': '信息'
      }
      return map[confidence] || confidence
    },

    getRiskType(risk) {
      const map = {
        'critical': 'danger',
        'high': 'danger',
        'medium': 'warning',
        'low': 'success',
        'info': 'info'
      }
      return map[risk] || 'info'
    }
  }
}
</script>

<style scoped>
.node-detail-panel {
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

.description-text {
  font-size: 14px;
  color: #606266;
  line-height: 1.6;
  background: #f8f9fa;
  padding: 12px;
  border-radius: 8px;
  border-left: 3px solid #409EFF;
}

.mitre-info {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.indicators-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}

.indicator-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metadata-grid {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.metadata-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 滚动条样式 */
.node-detail-panel::-webkit-scrollbar {
  width: 6px;
}

.node-detail-panel::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.node-detail-panel::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.node-detail-panel::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
