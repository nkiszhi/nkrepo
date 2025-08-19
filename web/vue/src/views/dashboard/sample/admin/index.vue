<template>
  <div class="dashboard-editor-container">
    <panel-group @handleSetLineChartData="handleSetLineChartData" />

    <el-row style="background:#fff;padding:16px 16px 0;margin-bottom:32px;">
      <line-chart :chart-data="lineChartData" />
    </el-row>

    <el-row :gutter="32">
      <el-col :xs="24" :sm="24" :lg="8">
        <div class="chart-wrapper">
          <pie1-chart /> <!-- 可展示category数据 -->
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :lg="8">
        <div class="chart-wrapper">
          <pie-chart /> <!-- 可展示family数据 -->
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :lg="8">
        <div class="chart-wrapper">
          <bar-chart /> <!-- 可展示platform数据 -->
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import GithubCorner from '@/components/GithubCorner'
import PanelGroup from './components/PanelGroup'
import LineChart from './components/LineChart'
import PieChart from './components/PieChart'
import Pie1Chart from './components/Pie1Chart'
import BarChart from './components/BarChart'

// 导入生成的JS数据文件（核心修改）
import chartData from '@/data/chart_data.js'

export default {
  name: 'DashboardAdmin',
  components: {
    GithubCorner,
    PanelGroup,
    LineChart,
    PieChart,
    Pie1Chart,
    BarChart
  },
  data() {
    return {
      // 从JS文件中读取折线图数据（默认显示总年份数据）
      lineChartData: chartData.lineChartData.total_amount,
      // 可将饼图数据通过props传递给子组件（按需使用）
      pieData: chartData.pieTop10Data
    }
  },
  methods: {
    // 切换折线图数据（从JS文件中读取）
    handleSetLineChartData(type) {
      this.lineChartData = chartData.lineChartData[type]
    }
  }
}
</script>

<style lang="scss" scoped>
/* 样式保持不变 */
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
}

@media (max-width:1024px) {
  .chart-wrapper {
    padding: 8px;
  }
}
</style>