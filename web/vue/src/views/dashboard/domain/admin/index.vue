<template>
  <div class="dashboard-editor-container">

    <panel-group @handleSetLineChartData="handleSetLineChartData" />

    <el-row style="background:#fff;padding:16px 16px 0;margin-bottom:32px;">
      <line-chart :chart-data="lineChartData" />
    </el-row>

    <el-row :gutter="32">
      <el-col :xs="24" :sm="24" :lg="12">
        <div class="chart-wrapper">
          <pie-chart />
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :lg="12">
        <div class="chart-wrapper">
          <bar-chart />
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import PanelGroup from './components/PanelGroup'
import LineChart from './components/LineChart'
import PieChart from './components/PieChart'
import BarChart from './components/BarChart'

const lineChartData = {
  total_domain: {
    date_data: ["2019年","2020年","2021年","2022年","2023年","2024年","2025年"],
    amount_data: [468.2,4001.2,1300.4,17830.6,19714.0,34513.4,14265.5]},
  messages: {
    date_data: ["2024年01月","2024年02月","2024年03月","2024年04月","2024年05月","2024年06月","2024年07月","2024年08月","2024年09月","2024年10月","2024年11月","2024年12月"],
    amount_data: [2407.2,2317.7,2563,2511.1,2854.6,2817.4,3224.3,3328.8,3363.1,3072.9,2809.6,2062.2]},
  purchases: {
    date_data: ["12月01日","12月02日","12月03日","12月04日","12月05日","12月06日","12月07日","12月08日","12月09日","12月10日","12月11日","12月12日","12月13日","12月14日","12月15日","12月16日","12月17日","12月18日","12月19日","12月20日","12月21日","12月22日","12月23日","12月24日","12月25日","12月26日","12月27日","12月28日","12月29日","12月30日","12月31日"],
    amount_data: [112.5,113.6,113.3,113.3,112.6,113.6,114.5,117.6,113.4,118.2,114.9,114.8,114.8,113.6,113.7,118.5,114.0,114.2,114.1,114.9,115,115.1,115.5,115.6,115.8,114.9,115.0,114.9,115.5,115.4,115.5]
  }
}

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
      lineChartData: lineChartData.total_domain
    }
  },
  methods: {
    handleSetLineChartData(type) {
      this.lineChartData = lineChartData[type]
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
}

@media (max-width:1024px) {
  .chart-wrapper {
    padding: 8px;
  }
}
</style>
