<template>
  <el-row :gutter="40" class="panel-group">
    <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('total_amount')">
        <div class="card-panel-icon-wrapper icon-people">
          <svg-icon icon-class="chart" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-info flex-container">  
            <span class="card-panel-text">总数量：</span>  
            <!-- 动态显示总数量（转换为万条） -->
            <count-to 
              :start-val="0" 
              :end-val="totalCount" 
              :duration="2600" 
              class="card-panel-num" 
            />  
            <span class="card-panel-text">万条</span>  
          </div>  
          <div class="card-panel-text">
            统计日期：{{ dateRange.total }}
          </div>
        </div>
      </div>
    </el-col>
    <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('year_amount')">
        <div class="card-panel-icon-wrapper icon-message">
          <svg-icon icon-class="chart" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-info flex-container">  
            <span class="card-panel-text">近一年数量：</span>  
            <!-- 动态显示近一年数量（转换为万条） -->
            <count-to 
              :start-val="0" 
              :end-val="recentCount" 
              :duration="2600" 
              class="card-panel-num" 
            />  
            <span class="card-panel-text">万条</span>  
          </div>  
          <div class="card-panel-text">
            统计日期：{{ dateRange.recent }}
          </div>
        </div>
      </div>
    </el-col>
  </el-row>
</template>

<script>
import CountTo from 'vue-count-to'
// 导入生成的JS数据文件
import chartData from '@/data/chart_data.js'

export default {
  components: {
    CountTo
  },
  data() {
    return {
      // 从JS数据中获取统计信息
      summary: chartData.summary,
      // 年份数据（用于计算日期范围）
      yearData: chartData.lineChartData.total_amount.date_data
    }
  },
  computed: {
    // 总数量（转换为万条，保留1位小数）
    totalCount() {
      return (this.summary.total_samples / 10000).toFixed(1)
    },
    // 近一年数量（转换为万条，保留1位小数）
    recentCount() {
      return (this.summary.recent_year_samples / 10000).toFixed(1)
    },
    // 日期范围计算
    dateRange() {
      // 总日期范围：取年份数据的第一个和最后一个
      const firstYear = this.yearData[0]?.replace('年', '') || ''
      const lastYear = this.yearData[this.yearData.length - 1]?.replace('年', '') || ''
      
      // 近一年日期范围：当前年份的1月到12月
      const currentYear = this.summary.current_year
      
      return {
        total: firstYear && lastYear ? `${firstYear}年1月-${lastYear}年12月` : '暂无数据',
        recent: `${currentYear}年1月-${currentYear}年12月`
      }
    }
  },
  methods: {
    handleSetLineChartData(type) {
      this.$emit('handleSetLineChartData', type)
    }
  }
}
</script>

<style lang="scss" scoped>
/* 样式保持不变 */
.panel-group {
  margin-top: 18px;

  .card-panel-col {
    margin-left: 10%;
    margin-right: 10%;
    margin-bottom: 30px;
  }

  .card-panel {
    height: 108px;
    width: 150%;
    cursor: pointer;
    font-size: 12px;
    position: relative;
    overflow: hidden;
    color: #666;
    background: #fff;
    box-shadow: 4px 4px 40px rgba(0, 0, 0, .05);
    border-color: rgba(0, 0, 0, .05);

    &:hover {
      .card-panel-icon-wrapper {
        color: #fff;
      }

      .icon-people {
        background: #40c9c6;
      }

      .icon-message {
        background: #36a3f7;
      }

    }

    .icon-people {
      color: #40c9c6;
    }

    .icon-message {
      color: #36a3f7;
    }


    .card-panel-icon-wrapper {
      float: left;
      margin: 14px 0 0 14px;
      padding: 16px;
      transition: all 0.38s ease-out;
      border-radius: 6px;
    }

    .card-panel-icon {
      float: left;
      font-size: 48px;
    }

    .flex-container {  
      display: flex;  
      align-items: center; /* 垂直居中 */  
    }  
  
    .card-panel-text {  
      margin-right: 4px; /* 根据需要调整间距 */  
    }  

    .card-panel-description {  
  font-weight: bold;  
  margin: 26px 10%; /* 简化margin设置 */  
  
  .card-panel-text,  
  .card-panel-num {  
    line-height: 18px;  
    color: rgba(0, 0, 0, 0.45);  
    font-size: 0.8vw; /* 保持响应式字体 */  
    margin-bottom: 12px;  
  }  
    }
  }
}

@media (max-width:550px) {
  .card-panel-description {
    display: none;
  }

  .card-panel-icon-wrapper {
    float: none !important;
    width: 100%;
    height: 100%;
    margin: 0 !important;

    .svg-icon {
      display: block;
      margin: 14px auto !important;
      float: none !important;
    }
  }
}
</style>