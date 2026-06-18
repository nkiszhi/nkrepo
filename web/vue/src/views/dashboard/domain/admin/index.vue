<template>
  <div class="dashboard-editor-container">

    <domain-summary-panels @handleSetLineChartData="handleSetLineChartData" />

    <el-row class="chart-panel chart-panel--line">
      <!-- 添加标题区域 -->
      <div class="chart-header chart-header--line">
        <h3 class="chart-title">{{ lineChartTitle }}</h3>
        <div class="chart-subtitle">{{ lineChartSubtitle }}</div>
      </div>
      <line-domain-trend :chart-data="lineChartData" height="clamp(260px, 42vh, 360px)" />
    </el-row>

    <el-row :gutter="24" class="chart-row">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper">
          <div class="chart-header">
            <h3 class="chart-title">恶意域名来源(Source)Top10</h3>
          </div>
          <div class="chart-body">
            <bar-domain-source :chart-data="sourceChartData" height="clamp(240px, 46vh, 420px)" />
          </div>
        </div>
      </el-col>
    </el-row>

    <el-row :gutter="24" class="chart-row">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper">
          <div class="chart-header">
            <h3 class="chart-title">恶意域名类型(Category)Top10</h3>
          </div>
          <div class="chart-body">
            <bar-domain-category :chart-data="categoryChartData" height="clamp(240px, 46vh, 420px)" />
          </div>
        </div>
      </el-col>
    </el-row>

  </div>
</template>

<script>
import DomainSummaryPanels from './components/domain-summary-panels.vue'
import LineDomainTrend from './components/line-domain-trend.vue'
import BarDomainSource from './components/bar-domain-source.vue'
import BarDomainCategory from './components/bar-domain-category.vue'

// 从chart_data.js导入数据
import chartData from '@/data/chart_data.js'

const lineChartData = chartData.lineChartDataDomain
const sourceChartData = chartData.top10Source || []
const categoryChartData = chartData.top10Category || []

export default {
  name: 'DashboardAdmin',
  components: {
    DomainSummaryPanels,
    LineDomainTrend,
    BarDomainSource,
    BarDomainCategory
  },
  data() {
    return {
      lineChartData: lineChartData.total_domain,
      sourceChartData,
      categoryChartData,
      currentChartType: 'total_domain'
    }
  },
  computed: {
    lineChartTitle() {
      return lineChartData[this.currentChartType]?.name || '恶意域名数量统计'
    },
    lineChartSubtitle() {
      return lineChartData[this.currentChartType]?.subtitle || ''
    }
  },
  methods: {
    handleSetLineChartData(type) {
      this.lineChartData = lineChartData[type]
      this.currentChartType = type
    }
  }
}
</script>

<style lang="scss" scoped>
.dashboard-editor-container {
  padding: 32px;
  background-color: rgb(240, 242, 245);
  position: relative;
  min-width: 0;

  .github-corner {
    position: absolute;
    top: 0px;
    border: 0;
    right: 0;
  }

  .chart-panel,
  .chart-wrapper {
    background: #fff;
    border-radius: 4px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  }

  .chart-panel {
    padding: 16px 16px 0;
    margin-bottom: 32px;
  }

  .chart-row {
    margin-bottom: 24px;
  }

  .chart-wrapper {
    padding: 12px;
    margin-bottom: 32px;
  }

  .chart-body {
    min-width: 0;
  }

  .chart-header {
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #ebeef5;

    h3.chart-title {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
      color: #333;
      line-height: 1.4;
    }

    .chart-subtitle {
      margin-top: 4px;
      font-size: 14px;
      color: #666;
      line-height: 1.4;
    }
  }

  .chart-header--line {
    margin-bottom: 16px;
    padding-bottom: 12px;
  }
}

@media (max-width: 768px) {
  .dashboard-editor-container {
    padding: 12px;

    .chart-panel,
    .chart-wrapper {
      padding: 12px;
      margin-bottom: 16px;
    }

    .chart-header h3.chart-title {
      font-size: 15px;
      line-height: 1.35;
    }

    .chart-header .chart-subtitle {
      font-size: 12px;
    }
  }
}
</style>
