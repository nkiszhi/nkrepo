<template>
  <div class="dashboard-editor-container">
    <panel-group @handleSetLineChartData="handleSetLineChartData" />

    <!-- 折线图区域 -->
    <el-row style="background:#fff;padding:16px 16px 0;margin-bottom:32px; min-height: 400px; border-radius: 4px; box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);">
      <!-- 添加标题区域 -->
      <div class="chart-header" style="margin-bottom: 16px; padding-bottom: 12px;">
        <h3 class="chart-title">{{ lineChartTitle }}</h3>
      </div>
      <line-chart :chart-data="lineChartData" />
    </el-row>

    <!-- 恶意文件样本类型top10 -->
    <el-row :gutter="32" style="margin-bottom: 24px;">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper" style="height: 450px;">
          <div class="chart-header" style="margin-bottom: 8px;">
            <h3 class="chart-title">恶意文件类型(Category)Top10</h3>
          </div>
          <div style="height: calc(100% - 40px);">
            <pie1-chart />
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- Family 饼图 -->
    <el-row :gutter="32" style="margin-bottom: 24px;">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper" style="height: 450px;">
          <div class="chart-header" style="margin-bottom: 8px;">
            <h3 class="chart-title">恶意文件平台(Platform)Top10</h3>
          </div>
          <div style="height: calc(100% - 40px);">
            <pie-chart />
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- Platform 饼图 -->
    <el-row :gutter="32" style="margin-bottom: 24px;">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper" style="height: 450px;">
          <div class="chart-header" style="margin-bottom: 8px;">
            <h3 class="chart-title">恶意文件家族(Family)Top10</h3>
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
import Pie1Chart from './components/Pie1Chart.vue'
import BarChart from './components/BarChart.vue'

import chartData from '@/data/chart_data.js'

export default {
  name: 'DashboardAdmin',
  components: {
    PanelGroup,
    LineChart,
    PieChart,
    Pie1Chart,
    BarChart
  },
  data() {
    return {
      lineChartData: chartData.lineChartData.total_amount,
      pieData: chartData.pieTop10Data,
      currentChartType: 'total_amount'
    }
  },
  computed: {
    lineChartTitle() {
      const titles = {
        'total_amount': this.getTotalAmountTitle(),
        'year_amount': this.getYearAmountTitle()
      }
      return titles[this.currentChartType] || '恶意文件数量统计'
    }
  },
  methods: {
    handleSetLineChartData(type) {
      this.lineChartData = chartData.lineChartData[type]
      this.currentChartType = type
    },
    getYearAmountTitle() {
      // 动态生成近一年标题
      const now = new Date()
      const currentYear = now.getFullYear()
      const currentMonth = now.getMonth() + 1 // getMonth()返回0-11
      
      // 计算起始年月（从当前月份往前推12个月）
      let startYear = currentYear - 1
      let startMonth = currentMonth + 1
      if (startMonth > 12) {
        startMonth = 1
        startYear = currentYear
      }
      
      return `${startYear}年${startMonth}月至${currentYear}年${currentMonth}月恶意文件数量统计`
    },
    getTotalAmountTitle() {
      // 动态生成年度标题
      const now = new Date()
      const currentYear = now.getFullYear()
      return `2012-${currentYear}年恶意文件总数统计`
    }
  }
}
</script>

<style lang="scss" scoped>
.dashboard-editor-container {
  padding: 32px;
  background-color: rgb(240, 242, 245);
  position: relative;

  .chart-wrapper {
    background: #fff;
    padding: 12px;
    border-radius: 4px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);

    .chart-header {
      padding: 0 0 8px 0;
      border-bottom: 1px solid #ebeef5;

      .chart-title {
        margin: 0;
        font-size: 16px;
        font-weight: 600;
        color: #303133;
      }
    }
  }

  .el-row:first-of-type {
    .chart-header {
      .chart-title {
        margin: 0;
        font-size: 18px;
        font-weight: 600;
        color: #303133;
        line-height: 1.4;
      }
    }
  }
}
</style>
