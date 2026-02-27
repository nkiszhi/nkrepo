<template>
  <el-drawer
    title="节点详情"
    v-model="visible"
    direction="rtl"
    size="30%"
    :before-close="handleClose"
    :wrapper-closable="false"
  >
    <div v-if="node" class="node-details-content">
      <!-- 基本信息 -->
      <el-card shadow="never" style="margin-bottom: 20px;">
        <div slot="header" class="card-header">
          <span style="font-size: 16px; font-weight: bold;">基本信息</span>
          <el-tag
            :type="getNodeTypeTag(node.type)"
            size="small"
            style="margin-left: 10px;"
          >
            {{ getNodeTypeLabel(node.type) }}
          </el-tag>
        </div>

        <el-descriptions :column="1" border size="small">
          <el-descriptions-item label="节点ID">
            <el-tag size="small" type="info">{{ node.id }}</el-tag>
          </el-descriptions-item>

          <el-descriptions-item label="节点名称">
            <strong>{{ (node.data && node.data.label) || (node.data && node.data.name) || '未命名' }}</strong>
          </el-descriptions-item>

          <el-descriptions-item v-if="node.data && node.data.confidence" label="置信度">
            <el-tag
              :type="getConfidenceTag(node.data.confidence)"
              size="small"
            >
              {{ getConfidenceLabel(node.data.confidence) }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>
      </el-card>

      <!-- 描述信息 -->
      <el-card v-if="node.data && node.data.description" shadow="never" style="margin-bottom: 20px;">
        <div slot="header">
          <span style="font-size: 16px; font-weight: bold;">描述</span>
        </div>
        <div style="line-height: 1.6; color: #606266;">
          {{ node.data.description }}
        </div>
      </el-card>

      <!-- MITRE ATT&CK信息 -->
      <el-card v-if="hasMitreInfo" shadow="never" style="margin-bottom: 20px;">
        <div slot="header">
          <span style="font-size: 16px; font-weight: bold;">MITRE ATT&CK</span>
        </div>
        
        <div v-if="node.data && node.data.technique_id" style="margin-bottom: 15px;">
          <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #303133;">技术ID</h4>
          <el-tag
            size="medium"
            type="danger"
            style="margin-bottom: 10px; font-size: 13px;"
          >
            {{ node.data.technique_id }}
          </el-tag>
          <div v-if="node.data.technique_name" style="font-size: 13px; color: #606266; margin-top: 5px;">
            技术名称: {{ node.data.technique_name }}
          </div>
        </div>

        <div v-if="node.data && node.data.tactic" style="margin-bottom: 15px;">
          <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #303133;">攻击战术</h4>
          <el-tag
            size="medium"
            type="warning"
            style="margin-bottom: 10px;"
          >
            {{ node.data.tactic }}
          </el-tag>
        </div>

        <div v-if="hasSubTechniques" style="margin-bottom: 15px;">
          <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #303133;">子技术</h4>
          <div style="display: flex; flex-wrap: wrap; gap: 5px;">
            <el-tag
              v-for="sub in node.data.sub_techniques"
              :key="sub"
              size="small"
              type="info"
            >
              {{ sub }}
            </el-tag>
          </div>
        </div>
      </el-card>

      <!-- 威胁指标 -->
      <el-card v-if="hasThreatIndicators" shadow="never" style="margin-bottom: 20px;">
        <div slot="header">
          <span style="font-size: 16px; font-weight: bold;">威胁指标</span>
        </div>

        <div v-if="node.data && node.data.indicators && node.data.indicators.length > 0">
          <h4 style="margin: 0 0 10px 0; font-size: 14px;">威胁指标</h4>
          <ul style="margin: 0; padding-left: 20px;">
            <li v-for="(indicator, index) in node.data.indicators" :key="index" style="margin-bottom: 5px; font-size: 13px;">
              {{ indicator }}
            </li>
          </ul>
        </div>

        <div v-if="node.data && node.data.iocs && node.data.iocs.length > 0" style="margin-top: 15px;">
          <h4 style="margin: 0 0 10px 0; font-size: 14px;">IoC指标</h4>
          <div style="display: flex; flex-wrap: wrap; gap: 5px;">
            <el-tag
              v-for="ioc in node.data.iocs"
              :key="ioc"
              size="small"
              type="info"
            >
              {{ ioc }}
            </el-tag>
          </div>
        </div>
      </el-card>

      <!-- 技术指标 -->
      <el-card v-if="hasTechnicalIndicators" shadow="never" style="margin-bottom: 20px;">
        <div slot="header">
          <span style="font-size: 16px; font-weight: bold;">技术指标</span>
        </div>

        <el-descriptions :column="1" size="small">
          <template v-for="(value, key) in technicalFields">
            <el-descriptions-item
              v-if="value"
              :key="key"
              :label="formatTechnicalKey(key)"
            >
              <span v-if="key === 'hash' || key === 'ip_address' || key === 'domain'" class="technical-value code">
                {{ value }}
              </span>
              <span v-else>
                {{ value }}
              </span>
            </el-descriptions-item>
          </template>
        </el-descriptions>
      </el-card>

      <!-- 元数据 -->
      <el-card v-if="hasMetadata" shadow="never">
        <div slot="header">
          <span style="font-size: 16px; font-weight: bold;">元数据</span>
        </div>

        <el-descriptions :column="1" size="small">
          <template v-for="(value, key) in metadataFields">
            <el-descriptions-item
              v-if="value"
              :key="key"
              :label="formatKey(key)"
            >
              <span v-if="typeof value === 'string'">
                {{ value }}
              </span>
              <span v-else-if="typeof value === 'object'">
                {{ JSON.stringify(value) }}
              </span>
              <span v-else>
                {{ value }}
              </span>
            </el-descriptions-item>
          </template>
        </el-descriptions>
      </el-card>

      <!-- 空状态 -->
      <div v-else class="no-data">
        <el-empty description="暂无详细数据" :image-size="100" />
      </div>
      
      <!-- 操作按钮 -->
      <div class="action-buttons">
        <el-button type="primary" size="small" @click="copyNodeInfo">
          <i class="el-icon-document-copy" style="margin-right: 5px;" />
          复制信息
        </el-button>
        <el-button size="small" @click="exportNode">
          <i class="el-icon-download" style="margin-right: 5px;" />
          导出节点
        </el-button>
      </div>
    </div>
  </el-drawer>
</template>

<script>
export default {
  name: 'FlowNodeDetails',
  props: {
    node: {
      type: Object,
      default: null
    },
    visible: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      excludedKeys: ['id', 'type', 'label', 'title', 'description', 'tactics', 'technique_id', 'technique_name', 
                    'tactic', 'sub_techniques', 'indicators', 'iocs', 'confidence', 'source_excerpt', 'created_at'],
      technicalKeys: ['ip_address', 'domain', 'url', 'file_path', 'process_name', 'command_line', 'hash', 
                     'registry_key', 'port', 'protocol', 'cve_id', 'cwe_id']
    }
  },

  computed: {
    hasMitreInfo() {
      return this.node && this.node.data && (
        this.node.data.technique_id || 
        this.node.data.tactic
      )
    },
    
    hasSubTechniques() {
      return this.node && this.node.data && 
             this.node.data.sub_techniques && 
             this.node.data.sub_techniques.length > 0
    },

    hasThreatIndicators() {
      return (
        (this.node && this.node.data && this.node.data.indicators && this.node.data.indicators.length > 0) ||
        (this.node && this.node.data && this.node.data.iocs && this.node.data.iocs.length > 0)
      )
    },
    
    hasTechnicalIndicators() {
      if (!this.node || !this.node.data) return false
      
      return this.technicalKeys.some(key => {
        return this.node.data[key] !== undefined && this.node.data[key] !== ''
      })
    },
    
    technicalFields() {
      if (!this.node || !this.node.data) return {}
      
      const fields = {}
      this.technicalKeys.forEach(key => {
        if (this.node.data[key] !== undefined && this.node.data[key] !== '') {
          fields[key] = this.node.data[key]
        }
      })
      return fields
    },

    hasMetadata() {
      if (!this.node || !this.node.data) return false

      return Object.keys(this.node.data).some(key => {
        return !this.excludedKeys.includes(key) && 
               !this.technicalKeys.includes(key) && 
               this.node.data[key]
      })
    },
    
    metadataFields() {
      if (!this.node || !this.node.data) return {}
      
      const fields = {}
      Object.keys(this.node.data).forEach(key => {
        if (!this.excludedKeys.includes(key) && 
            !this.technicalKeys.includes(key) && 
            this.node.data[key]) {
          fields[key] = this.node.data[key]
        }
      })
      return fields
    }
  },

  methods: {
    getNodeTypeTag(type) {
      const map = {
        'action': 'primary',
        'tool': 'warning',
        'malware': 'danger',
        'asset': 'success',
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
    
    formatKey(key) {
      return key
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .trim()
    },
    
    formatTechnicalKey(key) {
      const map = {
        'ip_address': 'IP地址',
        'domain': '域名',
        'url': 'URL',
        'file_path': '文件路径',
        'process_name': '进程名',
        'command_line': '命令行',
        'hash': '哈希值',
        'registry_key': '注册表键',
        'port': '端口',
        'protocol': '协议',
        'cve_id': 'CVE ID',
        'cwe_id': 'CWE ID'
      }
      return map[key] || this.formatKey(key)
    },

    handleClose() {
      this.$emit('update:visible', false)
      this.$emit('close')
    },
    
    copyNodeInfo() {
      if (!this.node) return
      
      const nodeInfo = {
        '节点ID': this.node.id,
        '节点类型': this.getNodeTypeLabel(this.node.type),
        '节点名称': this.node.data.label || this.node.data.name || '未命名',
        '描述': this.node.data.description || '无',
        '置信度': this.getConfidenceLabel(this.node.data.confidence || 'medium')
      }
      
      if (this.node.data.technique_id) {
        nodeInfo['MITRE技术ID'] = this.node.data.technique_id
      }
      
      if (this.node.data.tactic) {
        nodeInfo['MITRE战术'] = this.node.data.tactic
      }
      
      const text = Object.entries(nodeInfo)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\n')
      
      navigator.clipboard.writeText(text)
        .then(() => {
          this.$message.success('节点信息已复制到剪贴板')
        })
        .catch(err => {
          console.error('复制失败:', err)
          this.$message.error('复制失败')
        })
    },
    
    exportNode() {
      if (!this.node) return
      
      const nodeData = {
        node: this.node,
        exportedAt: new Date().toISOString(),
        format: 'FlowViz Node Export'
      }

      const dataStr = JSON.stringify(nodeData, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)

      const link = document.createElement('a')
      link.download = `node-${this.node.id}-${Date.now()}.json`
      link.href = url
      link.click()

      URL.revokeObjectURL(url)

      this.$message.success('节点信息已导出')
    }
  }
}
</script>

<style scoped>
.node-details-content {
  padding: 0 10px;
  height: 100%;
  overflow-y: auto;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.no-data {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
}

.technical-value.code {
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  background: #f5f5f5;
  padding: 2px 6px;
  border-radius: 4px;
  word-break: break-all;
}

.action-buttons {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #ebeef5;
  display: flex;
  gap: 10px;
  justify-content: center;
}

/* 滚动条样式 */
.node-details-content::-webkit-scrollbar {
  width: 6px;
}

.node-details-content::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.node-details-content::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.node-details-content::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>