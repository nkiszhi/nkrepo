<template>
  <div class="section">
    <div v-if="ensembleResult" class="model-summary">
      <span class="summary-label">多模型结果：</span>
      <span :class="['summary-result', ensembleResult.result === '恶意' ? 'malicious' : 'safe']">
        {{ ensembleResult.result }}
      </span>
      <span v-if="ensembleResult.probability !== null && ensembleResult.probability !== undefined" class="summary-probability">
        恶意概率 {{ formatProbability(ensembleResult.probability) }}
      </span>
      <span v-if="ensembleResult.virus_name" class="summary-virus">
        病毒名称：{{ ensembleResult.virus_name }}
      </span>
    </div>
    <table class="detection-result-table">
      <thead>
        <tr>
          <th>检测模型</th>
          <th>恶意概率</th>
          <th>结果</th>
          <th>病毒名称</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(resultData,model) in modelResults" :key="model">
          <td>{{ model }}</td>
          <td>{{ formatProbability(resultData.probability) }}</td>
          <td :style="{ color: resultData.result === '恶意' ? 'red' : 'inherit' }">
            {{ resultData.result }}
          </td>
          <td>{{ resultData.virus_name || '-' }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script>
export default {
  name: 'ModelDetection',
  props: {
    uploadResult: {
      type: Object,
      required: true
    }
  },
  computed: {
    allResults() {
      return this.uploadResult.exe_result || this.uploadResult.results || {}
    },
    ensembleResult() {
      return this.allResults['集成结果'] || null
    },
    modelResults() {
      return Object.fromEntries(
        Object.entries(this.allResults).filter(([model]) => model !== '集成结果' && model !== 'error')
      )
    }
  },
  methods: {
    formatProbability(probability) {
      if (probability === null || probability === undefined || probability === '') return '-'
      const value = Number(probability)
      if (Number.isNaN(value)) return probability
      return `${(value * 100).toFixed(2)}%`
    }
  }
}
</script>

<style scoped>
.detection-result-table {
  width: 80%;
  margin: 0 auto;
  border: 1px solid #ccc;
  border-collapse: collapse;
  margin-top: 30px;
}

.model-summary {
  width: 80%;
  margin: 20px auto 0;
  padding: 12px 16px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  background: #f8f9fb;
}

.summary-label {
  font-weight: 600;
}

.summary-result {
  margin-right: 12px;
  font-weight: 700;
}

.summary-result.malicious {
  color: #f56c6c;
}

.summary-result.safe {
  color: #67c23a;
}

.summary-probability,
.summary-virus {
  margin-left: 12px;
  color: #606266;
}

.detection-result-table th,
.detection-result-table td {
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.detection-result-table td:last-child {
  text-align: center;
}
</style>
