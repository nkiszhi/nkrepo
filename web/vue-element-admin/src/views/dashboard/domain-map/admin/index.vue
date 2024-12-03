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

    <!-- 新增的地图标题和地图组件区域 -->
    <el-row style="background:#fff;padding:16px;margin-top:32px;">
      <el-col :span="24">
        <h3>请求交互地图</h3>
        <map-chart />
      </el-col>
    </el-row>

  </div>
</template>

<script>
import GithubCorner from '@/components/GithubCorner'
import PanelGroup from './components/PanelGroup'
import LineChart from './components/LineChart'
import RaddarChart from './components/RaddarChart'
import PieChart from './components/PieChart'
import BarChart from './components/BarChart'
import TransactionTable from './components/TransactionTable'
import TodoList from './components/TodoList'
import BoxCard from './components/BoxCard'
import MapChart from './components/map'

const lineChartData = {
  total_domain: {
    date_data: ["2019年","2020年","2021年","2022年","2023年","2024年"],
    amount_data: [468.2,4001.2,1300.4,17637.4,20473.4,12487.0]},
  messages: {
    date_data: ["2023年07月","2023年08月","2023年09月","2023年10月","2023年11月","2023年12月","2024年01月","2024年02月","2024年03月","2024年04月","2024年05月"],
    amount_data: [139.8, 2003.1, 1851.0, 2302.8, 2103.9, 2238.7, 2407.2, 2317.7, 2481.6, 2511.1, 2769.4, ]},
  purchases: {
    date_data: ["5月01日","5月02日","5月03日","5月04日","5月05日","5月06日","5月07日","5月08日","5月09日","5月10日","5月11日","5月12日","5月13日","5月14日","5月15日","5月16日","5月17日","5月18日","5月19日","5月20日","5月21日","5月22日","5月23日","5月24日","5月25日","5月26日","5月27日","5月28日","5月29日","5月30日","5月31日"],
    amount_data: [90.0,91.9, 92.3, 92.4, 92.7, 89.9, 92.5, 93.2, 93.3, 92.8, 92.3, 92.3, 93.7, 93.7, 93.6, 92.6, 83.8, 83.8,92.6, 92.8, 94.0, 94.2, 94.1, 94.3, 93.1, 94.2, 94.2, 92.7, 92.8, 93.1, 93.4 ]
  }
}

export default {
  name: 'DashboardAdmin',
  components: {
    GithubCorner,
    PanelGroup,
    LineChart,
    RaddarChart,
    PieChart,
    BarChart,
    TransactionTable,
    TodoList,
    BoxCard,
    MapChart
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
