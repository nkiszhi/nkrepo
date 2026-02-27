<!-- vue/src/views/flowviz/components/FlowProgress.vue -->
<template>
  <div v-if="progress.message" class="flow-progress">
    <div class="progress-card">
      <div class="progress-header">
        <i v-if="progress.status === ''" class="el-icon-loading" />
        <i v-else-if="progress.status === 'success'" class="el-icon-success" />
        <i v-else-if="progress.status === 'exception'" class="el-icon-error" />
        <span class="progress-title">分析进度</span>
      </div>

      <div class="progress-content">
        <!-- 进度条 -->
        <el-progress
          :percentage="progress.percentage"
          :status="progress.status"
          :stroke-width="12"
          :text-inside="true"
          :format="progressFormat"
        />

        <!-- 消息 -->
        <div class="progress-message">
          {{ progress.message }}
        </div>

        <!-- 统计信息 -->
        <div v-if="stats" class="progress-stats">
          <div class="stat-item">
            <span class="stat-label">节点:</span>
            <span class="stat-value">{{ stats.nodes }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">边:</span>
            <span class="stat-value">{{ stats.edges }}</span>
          </div>
        </div>
      </div>

      <!-- 操作按钮 -->
      <div v-if="progress.status === ''" class="progress-actions">
        <el-button
          size="small"
          type="danger"
          :loading="cancelling"
          @click="$emit('cancel')"
        >
          取消分析
        </el-button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'FlowProgress',
  props: {
    progress: {
      type: Object,
      default: () => ({
        message: '',
        percentage: 0,
        status: '' // '', 'success', 'exception'
      })
    },
    stats: {
      type: Object,
      default: null
    }
  },
  data() {
    return {
      cancelling: false
    }
  },

  methods: {
    progressFormat(percentage) {
      if (percentage === 100) return '完成'
      return `${percentage}%`
    }
  }
}
</script>

<style scoped>
.flow-progress {
  position: absolute;
  top: 20px;
  right: 20px;
  z-index: 1000;
  max-width: 400px;
}

.progress-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  overflow: hidden;
  border: 1px solid #ebeef5;
}

.progress-header {
  background: linear-gradient(135deg, #409EFF 0%, #66b1ff 100%);
  color: white;
  padding: 12px 16px;
  display: flex;
  align-items: center;
}

.progress-header i {
  font-size: 18px;
  margin-right: 10px;
}

.progress-header .el-icon-success {
  color: #67c23a;
}

.progress-header .el-icon-error {
  color: #f56c6c;
}

.progress-title {
  font-size: 16px;
  font-weight: bold;
}

.progress-content {
  padding: 20px;
}

.progress-message {
  margin-top: 10px;
  font-size: 14px;
  color: #606266;
  text-align: center;
  min-height: 24px;
}

.progress-stats {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-top: 15px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}

.stat-value {
  font-size: 18px;
  font-weight: bold;
  color: #409EFF;
}

.progress-actions {
  padding: 10px 20px 20px;
  text-align: center;
  border-top: 1px solid #ebeef5;
}
</style>
