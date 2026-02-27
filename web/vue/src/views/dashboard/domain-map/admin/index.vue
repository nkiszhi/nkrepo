<template>
  <div class="dashboard-editor-container">

    <panel-group @handleSetLineChartData="handleSetLineChartData" />

    <el-row style="background:#fff;padding:16px 16px 0;margin-bottom:32px;">
      <!-- 添加标题区域 -->
      <div class="chart-header" style="margin-bottom: 16px;">
        <h3 class="chart-title">{{ lineChartTitle }}</h3>
        <div class="chart-subtitle">{{ lineChartSubtitle }}</div>
      </div>
      <line-chart :chart-data="lineChartData" />
    </el-row>

    <el-row :gutter="32" style="margin-bottom: 24px;">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper" style="height: 450px;">
          <div class="chart-header" style="margin-bottom: 8px;">
            <h3 class="chart-title">恶意域名来源(Source)Top10</h3>
          </div>
          <div style="height: calc(100% - 40px);">
            <pie-chart />
          </div>
        </div>
      </el-col>
    </el-row>

    <el-row :gutter="32" style="margin-bottom: 24px;">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper" style="height: 450px;">
          <div class="chart-header" style="margin-bottom: 8px;">
            <h3 class="chart-title">恶意域名类型(Category)Top10</h3>
          </div>
          <div style="height: calc(100% - 40px);">
            <bar-chart />
          </div>
        </div>
      </el-col>
    </el-row>

  </div>
</template>

<script>
import PanelGroup from './components/PanelGroup.vue'
import LineChart from './components/LineChart.vue'
import PieChart from './components/PieChart.vue'
import BarChart from './components/BarChart.vue'

// 从chart_data.js导入数据
import chartData from '@/data/chart_data.js'

// 使用导入的数据
const lineChartData = chartData.lineChartDataDomain

export default {
  name: 'DashboardAdmin',
  components: {
    PanelGroup,
    LineChart,
    PieChart,
    BarChart
  },
  data() {
    return {
      lineChartData: lineChartData.total_domain,
      currentChartType: 'total_domain',
      // 新增数据
      sourceData: chartData.top10Source || [],
      categoryData: chartData.top10Category || []
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

  .github-corner {
    position: absolute;
    top: 0px;
    border: 0;
    right: 0;
  }

  .chart-wrapper {
    background: #fff;
    padding: 16px 16px 0;
    margin-bottom: 32px;
  }

  .chart-header {
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
}

@media (max-width:1024px) {
  .chart-wrapper {
    padding: 8px;
  }
}
</style>
