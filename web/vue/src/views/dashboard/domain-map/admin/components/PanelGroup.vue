<template>
  <el-row :gutter="40" class="panel-group">
    <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('total_domain')">
        <div class="card-panel-icon-wrapper icon-people">
          <svg-icon icon-class="chart" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-info flex-container">
            <span class="card-panel-text card-panel-label">总数量：</span>
            <count-to
              v-if="panelData.total_domain.value > 0"
              :start-val="0"
              :end-val="panelData.total_domain.value"
              :duration="2600"
              class="card-panel-num"
            />
            <span v-else class="card-panel-num">--</span>
            <span class="card-panel-text card-panel-unit">万条</span>
          </div>
          <div class="card-panel-text card-panel-date">
            统计日期：{{ panelData.total_domain.dateRange || '--' }}
          </div>
        </div>
      </div>
    </el-col>
    <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('messages')">
        <div class="card-panel-icon-wrapper icon-message">
          <svg-icon icon-class="chart" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-info flex-container">
            <span class="card-panel-text card-panel-label">近一年数量：</span>
            <count-to
              v-if="panelData.messages.value > 0"
              :start-val="0"
              :end-val="panelData.messages.value"
              :duration="2600"
              class="card-panel-num"
            />
            <span v-else class="card-panel-num">--</span>
            <span class="card-panel-text card-panel-unit">万条</span>
          </div>
          <div class="card-panel-text card-panel-date">
            统计日期：{{ panelData.messages.dateRange || '--' }}
          </div>
        </div>
      </div>
    </el-col>
    <el-col :xs="12" :sm="12" :lg="6" class="card-panel-col">
      <div class="card-panel" @click="handleSetLineChartData('purchases')">
        <div class="card-panel-icon-wrapper icon-money">
          <svg-icon icon-class="chart" class-name="card-panel-icon" />
        </div>
        <div class="card-panel-description">
          <div class="card-panel-info flex-container">
            <span class="card-panel-text card-panel-label">近一个月数量：</span>
            <count-to
              v-if="panelData.purchases.value > 0"
              :start-val="0"
              :end-val="panelData.purchases.value"
              :duration="2600"
              class="card-panel-num"
            />
            <span v-else class="card-panel-num">--</span>
            <span class="card-panel-text card-panel-unit">万条</span>
          </div>
          <div class="card-panel-text card-panel-date">
            统计日期：{{ panelData.purchases.dateRange || '--' }}
          </div>
        </div>
      </div>
    </el-col>
  </el-row>
</template>

<script>
import CountTo from '@/components/CountTo/index.vue'
import chartData from '@/data/chart_data.js'

export default {
  name: 'PanelGroup',
  components: {
    CountTo
  },
  data() {
    return {
      panelData: {
        total_domain: {
          value: 0,
          dateRange: ''
        },
        messages: {
          value: 0,
          dateRange: ''
        },
        purchases: {
          value: 0,
          dateRange: ''
        }
      },
      rawData: null
    }
  },
  mounted() {
    this.initData()
  },
  methods: {
    handleSetLineChartData(type) {
      console.log('点击面板，传递类型:', type)
      this.$emit('handleSetLineChartData', type)
    },

    initData() {
      try {
        console.log('正在加载chart_data.js数据...')

        let lineChartDataDomain = null

        if (chartData.lineChartDataDomain) {
          lineChartDataDomain = chartData.lineChartDataDomain
        } else if (chartData.default && chartData.default.lineChartDataDomain) {
          lineChartDataDomain = chartData.default.lineChartDataDomain
        } else {
          lineChartDataDomain = chartData
        }

        console.log('获取到的数据:', lineChartDataDomain)
        this.rawData = lineChartDataDomain

        if (!lineChartDataDomain) {
          console.error('无法获取lineChartDataDomain数据')
          return
        }

        // 处理总数量数据
        if (lineChartDataDomain.total_domain) {
          const data = lineChartDataDomain.total_domain
          console.log('total_domain原始数据:', data)

          // 计算总数（转换为万条）
          const total = data.amount_data.reduce((sum, val) => sum + val, 0)
          const valueInWan = Math.round(total / 10000)

          // 生成日期范围 - 动态获取
          const dates = data.date_data
          let dateRange = ''
          if (dates.length > 0) {
            // 获取第一个年份和最后一个年份
            const extractYear = (dateStr) => {
              const yearMatch = dateStr.match(/(\d+)年/)
              return yearMatch ? yearMatch[1] : ''
            }

            const firstYear = extractYear(dates[0])
            const lastYear = extractYear(dates[dates.length - 1])

            if (firstYear && lastYear) {
              dateRange = `${firstYear}年-${lastYear}年`
            } else if (firstYear) {
              dateRange = `${firstYear}年`
            }
          }

          this.panelData.total_domain = {
            value: valueInWan,
            dateRange: dateRange
          }

          console.log('total_domain处理结果:', this.panelData.total_domain)
        }

        // 处理近一年数据
        if (lineChartDataDomain.messages) {
          const data = lineChartDataDomain.messages
          console.log('messages原始数据:', data)

          // 计算总数（转换为万条）
          const total = data.amount_data.reduce((sum, val) => sum + val, 0)
          const valueInWan = Math.round(total / 10000)

          // 生成日期范围 - 动态获取
          const dates = data.date_data
          let dateRange = ''
          if (dates.length > 0) {
            // 获取第一个和最后一个日期
            const firstDate = dates[0]
            const lastDate = dates[dates.length - 1]

            // 如果日期格式包含年份，直接使用
            if (firstDate.includes('年') && lastDate.includes('年')) {
              dateRange = `${firstDate}-${lastDate}`
            } else {
              // 否则只显示月份
              dateRange = `${firstDate}至${lastDate}`
            }
          }

          this.panelData.messages = {
            value: valueInWan,
            dateRange: dateRange
          }

          console.log('messages处理结果:', this.panelData.messages)
        }

        // 处理近一个月数据
        if (lineChartDataDomain.purchases) {
          const data = lineChartDataDomain.purchases
          console.log('purchases原始数据:', data)

          // 计算总数（转换为万条）
          const total = data.amount_data.reduce((sum, val) => sum + val, 0)
          const valueInWan = Math.round(total / 10000)

          // 生成日期范围 - 动态获取数据中的日期
          const dates = data.date_data
          let dateRange = ''
          if (dates.length > 0) {
            // 获取第一个和最后一个日期
            const firstDate = dates[0]
            const lastDate = dates[dates.length - 1]

            // 判断是否包含年份信息
            const hasYearInFirstDate = firstDate.includes('年')
            const hasYearInLastDate = lastDate.includes('年')

            if (hasYearInFirstDate && hasYearInLastDate) {
              // 如果日期中包含年份信息，直接使用
              dateRange = `${firstDate}-${lastDate}`
            } else {
              // 如果日期中没有年份信息，尝试获取当前年份
              const currentYear = new Date().getFullYear()

              // 尝试解析月份，判断是否跨年
              const parseMonth = (dateStr) => {
                const monthMatch = dateStr.match(/(\d+)月/)
                return monthMatch ? parseInt(monthMatch[1]) : 0
              }

              const firstMonth = parseMonth(firstDate)
              const lastMonth = parseMonth(lastDate)

              // 判断是否需要添加年份
              if (firstMonth <= lastMonth) {
                // 同一年内
                dateRange = `${currentYear}年${firstDate}-${lastDate}`
              } else {
                // 跨年份（如12月到1月）
                dateRange = `${currentYear - 1}年${firstDate}-${currentYear}年${lastDate}`
              }
            }
          }

          this.panelData.purchases = {
            value: valueInWan,
            dateRange: dateRange
          }

          console.log('purchases处理结果:', this.panelData.purchases)
        }

        console.log('最终面板数据:', this.panelData)
      } catch (error) {
        console.error('初始化数据时出错:', error)
      }
    }
  }
}
</script>

<style lang="scss" scoped>
.panel-group {
  margin-top: 18px;

  .card-panel-col {
    margin-left: 0%;
    margin-right: 8%;
    margin-bottom: 30px;
  }

  .card-panel {
    height: 110px;
    width: 120%;
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

      .icon-money {
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
      align-items: baseline; /* 改为基线对齐，让数字和文字在同一水平线上 */
      flex-wrap: nowrap;
      white-space: nowrap;
    }

    /* 修改1：分离不同类型的card-panel-text样式 */
    .card-panel-text {
      margin-right: 4px;
      font-size: 14px;
      color: rgba(0, 0, 0, 0.65);
      line-height: 22px;
      white-space: nowrap;
    }

    /* 修改2：为标签文字添加加粗 */
    .card-panel-label {
      font-weight: bold; /* 主要修改：加粗标签文字 */
      color: rgba(0, 0, 0, 0.65);

    }

    /* 修改3：为单位文字添加加粗 */
    .card-panel-unit {
      font-weight: 600; /* 可选：600比bold稍细，保持视觉平衡 */
      color: rgba(0, 0, 0, 0.75);
    }

    /* 修改4：为日期文字添加加粗 */
    .card-panel-date {
      font-weight: 600; /* 加粗 */
      color: rgba(0, 0, 0, 0.65); /* 保持原有颜色 */
      margin-top: 4px;
      font-size: 13px;
    }

    .card-panel-num {
      font-size: 22px;
      font-weight: bold;
      color: #333;
      margin: 0 4px;
      line-height: 22px;
    }

    .card-panel-description {
      font-weight: normal;
      margin: 26px 10%;

      /* 修改5：覆盖原有的card-panel-text样式 */
      .card-panel-text {
        line-height: 22px;
        font-size: 14px;
        margin-bottom: 0;
        font-weight: bold; /* 确保加粗 */
      }

      /* 修改6：统计日期的样式 */
      .card-panel-text:last-child {
        margin-top: 4px;
        font-size: 13px;
        color: rgba(0, 0, 0, 0.65);
        font-weight: 600; /* 加粗 */
      }
    }
  }
}

/* 修改7：响应式调整 - 保持加粗效果 */
@media (max-width: 768px) {
  .panel-group {
    .card-panel-description {
      .card-panel-text {
        font-size: 12px;
        font-weight: bold; /* 保持加粗 */
      }

      .card-panel-num {
        font-size: 18px;
        font-weight: bold; /* 保持加粗 */
      }

      .card-panel-label,
      .card-panel-unit,
      .card-panel-date {
        font-weight: bold; /* 保持加粗 */
      }
    }
  }
}

/* 修改8：超小屏幕调整 */
@media (max-width: 550px) {
  .card-panel-description {
    display: block !important;
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

  .panel-group {
    .card-panel-col {
      margin-left: 2%;
      margin-right: 2%;
    }

    .card-panel {
      .flex-container {
        flex-wrap: wrap;
      }

      .card-panel-description {
        .card-panel-text {
          font-size: 11px;
          font-weight: bold; /* 保持加粗 */
        }

        .card-panel-label,
        .card-panel-unit,
        .card-panel-date {
          font-size: 11px;
          font-weight: bold; /* 保持加粗 */
        }

        .card-panel-num {
          font-size: 16px;
          font-weight: bold; /* 保持加粗 */
        }
      }
    }
  }
}

/* 修改9：添加打印样式，确保打印时也加粗 */
@media print {
  .card-panel-label,
  .card-panel-unit,
  .card-panel-date,
  .card-panel-text {
    font-weight: bold !important;
    color: #000 !important;
  }
}
</style>
