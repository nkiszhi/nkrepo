<template>
  <div class="panel-group-container">
    <el-row :gutter="40" class="panel-group">
      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel" @click="handleSetLineChartData('total_amount')">
          <div class="card-panel-icon-wrapper icon-people">
            <svg-icon icon-class="chart" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-info flex-container">
              <span class="card-panel-text card-panel-label">总数量：</span>
              <count-to :start-val="0" :end-val="panelData.total_amount.value" :duration="2600" class="card-panel-num" />
              <span class="card-panel-unit">万条</span>
            </div>
            <div class="card-panel-date">{{ panelData.total_amount.dateRange }}</div>
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
              <span class="card-panel-text card-panel-label">近一年：</span>
              <count-to :start-val="0" :end-val="panelData.year_amount.value" :duration="3000" class="card-panel-num" />
              <span class="card-panel-unit">万条</span>
            </div>
            <div class="card-panel-date">{{ panelData.year_amount.dateRange }}</div>
          </div>
        </div>
      </el-col>

      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-money">
            <svg-icon icon-class="user" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-info flex-container">
              <span class="card-panel-text card-panel-label">良性样本：</span>
              <count-to :start-val="0" :end-val="panelData.benign_samples.value" :duration="3200" class="card-panel-num" />
              <span class="card-panel-unit">万条</span>
            </div>
            <div class="card-panel-date">{{ panelData.benign_samples.dateRange }}</div>
          </div>
        </div>
      </el-col>

      <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
        <div class="card-panel">
          <div class="card-panel-icon-wrapper icon-shopping">
            <svg-icon icon-class="bug" class-name="card-panel-icon" />
          </div>
          <div class="card-panel-description">
            <div class="card-panel-info flex-container">
              <span class="card-panel-text card-panel-label">恶意样本：</span>
              <count-to :start-val="0" :end-val="panelData.malicious_samples.value" :duration="3600" class="card-panel-num" />
              <span class="card-panel-unit">万条</span>
            </div>
            <div class="card-panel-date">{{ panelData.malicious_samples.dateRange }}</div>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import CountTo from '@/components/CountTo/index.vue'
import chartData from '@/data/chart_data.js'

export default {
  components: {
    CountTo
  },
  data() {
    return {
      panelData: {
        total_amount: { value: 0, dateRange: '' },
        year_amount: { value: 0, dateRange: '' },
        benign_samples: { value: 0, dateRange: '' },
        malicious_samples: { value: 0, dateRange: '' }
      }
    }
  },
  mounted() {
    this.initData()
  },
  methods: {
    handleSetLineChartData(type) {
      this.$emit('handleSetLineChartData', type)
    },
    
    initData() {
      try {
        console.log('正在加载chart_data.js数据...')
        
        if (!chartData) {
          console.error('数据格式错误，无法找到chartData字段')
          return
        }

        const summary = chartData.summary || {}
        const lineChartData = chartData.lineChartData || {}

        // 处理总数量数据
        if (summary.total_samples) {
          const valueInWan = Math.round(summary.total_samples / 10000)
          const currentYear = new Date().getFullYear()
          this.panelData.total_amount = {
            value: valueInWan,
            dateRange: `2012-${currentYear}年`
          }
        }

        // 处理近一年数量数据
        if (summary.recent_year_samples !== undefined) {
          const valueInWan = Math.round(summary.recent_year_samples / 10000)
          const now = new Date()
          const currentYear = now.getFullYear()
          const currentMonth = now.getMonth() + 1
          
          // 计算起始年月（从当前月份往前推12个月）
          let startYear = currentYear - 1
          let startMonth = currentMonth + 1
          if (startMonth > 12) {
            startMonth = 1
            startYear = currentYear
          }
          
          this.panelData.year_amount = {
            value: valueInWan,
            dateRange: `${startYear}年${startMonth}月至${currentYear}年${currentMonth}月`
          }
        }

        // 处理良性样本数据
        if (summary.benign_samples !== undefined) {
          const valueInWan = Math.round(summary.benign_samples / 10000)
          this.panelData.benign_samples = {
            value: valueInWan,
            dateRange: this.panelData.total_amount.dateRange
          }
        }

        // 处理恶意样本数据
        if (summary.malicious_samples !== undefined) {
          const valueInWan = Math.round(summary.malicious_samples / 10000)
          this.panelData.malicious_samples = {
            value: valueInWan,
            dateRange: this.panelData.total_amount.dateRange
          }
        }

        console.log('面板数据初始化完成:', this.panelData)
      } catch (error) {
        console.error('初始化数据时出错:', error)
      }
    }
  }
}
</script>

<style lang="scss" scoped>
.panel-group-container {
  padding: 20px 0;
}

.panel-group {
  margin-top: 18px;

  .card-panel-col {
    margin-bottom: 32px;
  }

  .card-panel {
    height: 108px;
    cursor: pointer;
    font-size: 12px;
    position: relative;
    overflow: hidden;
    color: #666;
    background: #fff;
    box-shadow: 4px 4px 40px rgba(0, 0, 0, 0.05);
    border-color: rgba(0, 0, 0, 0.05);
    border-radius: 8px;
    transition: all 0.3s;

    &:hover {
      transform: translateY(-5px);
      box-shadow: 0 7px 21px rgba(0, 0, 0, 0.1);
      
      .card-panel-icon-wrapper {
        color: #fff;
      }
      
      .icon-people {
        background: #40c9c6;
      }
      
      .icon-message {
        background: #36a3f7;
      }
      
      .icon-money {
        background: #34bfa3;
      }
      
      .icon-shopping {
        background: #f4516c;
      }
    }

    .icon-people {
      color: #40c9c6;
    }
    
    .icon-message {
      color: #36a3f7;
    }
    
    .icon-money {
      color: #34bfa3;
    }
    
    .icon-shopping {
      color: #f4516c;
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
      align-items: center;
    }

    .card-panel-text {
      margin-right: 0px;
    }

    .card-panel-description {
      font-weight: bold;
      margin: 26px 10%;

      .card-panel-info {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
      }

      .card-panel-text {
        line-height: 18px;
        color: rgba(0, 0, 0, 0.45);
        font-size: 0.79vw;
      }

      .card-panel-label {
        margin-right: 8px;
      }

      .card-panel-num {
        font-size: 24px;
        color: #333;
        margin: 0 4px;
      }

      .card-panel-unit {
        font-size: 14px;
        color: #666;
      }

      .card-panel-date {
        font-size: 12px;
        color: #999;
        margin-top: 4px;
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
