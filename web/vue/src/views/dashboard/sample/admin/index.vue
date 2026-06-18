<template>
  <div class="dashboard-editor-container">
    <sample-summary-panels
      :summary="chartData.summary"
      @handleSetLineChartData="handleSetLineChartData"
    />

    <!-- 折线图区域 -->
    <el-row class="chart-panel chart-panel--line">
      <!-- 添加标题区域 -->
      <div class="chart-header chart-header--line">
        <h3 class="chart-title">{{ lineChartTitle }}</h3>
      </div>
      <line-sample-trend v-if="chartDataReady" :key="`line-${resizeKey}`" :chart-data="lineChartData" :resize-key="resizeKey" height="clamp(260px, 42vh, 360px)" />
      <el-skeleton v-else :rows="6" animated />
    </el-row>

    <!-- 恶意文件样本类型top10 -->
    <el-row :gutter="24" class="chart-row">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper">
          <div class="chart-header">
            <h3 class="chart-title">恶意文件类型(Category)Top10</h3>
          </div>
          <div class="chart-body">
            <pie-category v-if="chartDataReady" :key="`category-${resizeKey}`" :chart-data="chartData.pieTop10Data.category" :resize-key="resizeKey" height="clamp(220px, 42vh, 400px)" />
            <el-skeleton v-else :rows="6" animated />
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- Family 饼图 -->
    <el-row :gutter="24" class="chart-row">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper">
          <div class="chart-header">
            <h3 class="chart-title">恶意文件平台(Platform)Top10</h3>
          </div>
          <div class="chart-body">
            <pie-platform v-if="chartDataReady" :key="`platform-${resizeKey}`" :chart-data="chartData.pieTop10Data.platform" :resize-key="resizeKey" height="clamp(220px, 42vh, 400px)" />
            <el-skeleton v-else :rows="6" animated />
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- Platform 饼图 -->
    <el-row :gutter="24" class="chart-row">
      <el-col :xs="24" :sm="24" :lg="24">
        <div class="chart-wrapper">
          <div class="chart-header">
            <h3 class="chart-title">恶意文件家族(Family)Top10</h3>
          </div>
          <div class="chart-body">
            <pie-family v-if="chartDataReady" :key="`family-${resizeKey}`" :chart-data="chartData.pieTop10Data.family" :resize-key="resizeKey" height="clamp(220px, 42vh, 400px)" />
            <el-skeleton v-else :rows="6" animated />
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { defineAsyncComponent } from 'vue'

const emptyChartData = {
  summary: {},
  lineChartData: {
    total_amount: { date_data: [], amount_data: [] },
    year_amount: { date_data: [], amount_data: [] }
  },
  pieTop10Data: {
    category: [],
    platform: [],
    family: []
  }
}

export default {
  name: 'DashboardAdmin',
  components: {
    SampleSummaryPanels: defineAsyncComponent(() => import('./components/sample-summary-panels.vue')),
    LineSampleTrend: defineAsyncComponent(() => import('./components/line-sample-trend.vue')),
    PiePlatform: defineAsyncComponent(() => import('./components/pie-platform.vue')),
    PieCategory: defineAsyncComponent(() => import('./components/pie-category.vue')),
    PieFamily: defineAsyncComponent(() => import('./components/pie-family.vue'))
  },
  data() {
    return {
      chartData: emptyChartData,
      chartDataReady: false,
      lineChartData: emptyChartData.lineChartData.total_amount,
      currentChartType: 'total_amount',
      resizeKey: 0,
      resizeTimer: null
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
  mounted() {
    this.loadChartData()
    window.addEventListener('resize', this.handleDashboardResize, { passive: true })
  },
  beforeUnmount() {
    window.removeEventListener('resize', this.handleDashboardResize)
    if (this.resizeTimer) {
      window.clearTimeout(this.resizeTimer)
      this.resizeTimer = null
    }
  },
  methods: {
    handleDashboardResize() {
      if (this.resizeTimer) {
        window.clearTimeout(this.resizeTimer)
      }
      this.resizeTimer = window.setTimeout(() => {
        this.resizeTimer = null
        this.resizeKey += 1
      }, 180)
    },
    async loadChartData() {
      try {
        const module = await import('@/data/chart_data.js')
        this.chartData = module.default || module
        this.lineChartData = this.chartData.lineChartData?.[this.currentChartType] || emptyChartData.lineChartData.total_amount
        this.chartDataReady = true
      } catch (error) {
        console.error('加载图表数据失败:', error)
      }
    },
    handleSetLineChartData(type) {
      this.lineChartData = this.chartData.lineChartData?.[type] || emptyChartData.lineChartData.total_amount
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
  min-width: 0;

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

    .chart-header {
      padding: 0 0 8px 0;
      margin-bottom: 8px;
      border-bottom: 1px solid #ebeef5;

      .chart-title {
        margin: 0;
        font-size: 16px;
        font-weight: 600;
        color: #303133;
      }
    }
  }

  .chart-body {
    min-width: 0;
    overflow: hidden;
  }

  .chart-header--line {
    margin-bottom: 16px;
    padding-bottom: 12px;

    .chart-title {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
      color: #303133;
      line-height: 1.4;
    }
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

    .chart-header--line .chart-title,
    .chart-wrapper .chart-header .chart-title {
      font-size: 15px;
      line-height: 1.35;
    }
  }
}
</style>
