# -*- coding: utf-8 -*-
import codecs

content = '''<template>
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
      currentChartType: 'total_amount' // 当前图表类型
    }
  },
  computed: {
    lineChartTitle() {
      // 根据当前图表类型返回对应的标题
      const titles = {
        'total_amount': '2012-2025年恶意文件总数统计',
        'year_amount': '2025年2月至2026年1月恶意文件数量统计'
      }
      return titles[this.currentChartType] || '恶意文件数量统计'
    }
  },
  methods: {
    handleSetLineChartData(type) {
      this.lineChartData = chartData.lineChartData[type]
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

  // 折线图区域的标题样式
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

  .github-corner {
    position: absolute;
    top: 0px;
    border: 0;
    right: 0;
  }
}

@media (max-width:1024px) {
  .dashboard-editor-container {
    padding: 16px;

    .chart-wrapper {
      padding: 8px;
      height: 380px !important;

      .chart-header {
        padding: 0 0 6px 0;

        .chart-title {
          font-size: 14px;
        }
      }

      div[style*="height: calc(100% - 40px)"] {
        height: calc(100% - 35px) !important;
      }
    }

    // 折线图区域移动端调整
    .el-row:first-of-type {
      min-height: 320px !important;

      .chart-header {
        .chart-title {
          font-size: 16px;
        }
      }
    }
  }
}

@media (max-width: 768px) {
  .dashboard-editor-container {
    padding: 12px;

    .chart-wrapper {
      height: 320px !important;
      margin-bottom: 16px !important;

      .chart-header {
        padding: 0 0 4px 0;
        margin-bottom: 6px !important;

        .chart-title {
          font-size: 13px;
        }
      }

      div[style*="height: calc(100% - 40px)"] {
        height: calc(100% - 30px) !important;
      }
    }

    .el-row:first-of-type {
      min-height: 250px !important;
      margin-bottom: 16px !important;
      padding: 12px 12px 0 !important;

      .chart-header {
        margin-bottom: 12px !important;
        padding-bottom: 8px !important;

        .chart-title {
          font-size: 14px;
        }
      }
    }
  }
}
</style>
'''

with codecs.open('src/views/dashboard/sample/admin/index.vue', 'w', 'utf-8') as f:
    f.write(content)

print('✅ admin/index.vue 已重写完成')
