<template>
  <div :class="className" :style="{height:height,width:width}" />
</template>

<script>
import * as echarts from 'echarts'
import resize from './mixins/resize.js'

const animationDuration = 6000

export default {
  mixins: [resize],
  props: {
    className: {
      type: String,
      default: 'chart'
    },
    width: {
      type: String,
      default: '100%'
    },
    height: {
      type: String,
      default: '400px'
    },
    chartData: {
      type: Array,
      default: () => []
    }
  },
  data() {
    return {
      chart: null
    }
  },
  watch: {
    chartData() {
      this.setOptions()
    }
  },
  mounted() {
    this.$nextTick(this.initChart)
  },
  beforeUnmount() {
    if (!this.chart) return
    this.chart.dispose()
    this.chart = null
  },
  methods: {
    initChart() {
      this.chart = echarts.init(this.$el, 'macarons')
      this.setOptions()
    },
    setOptions() {
      if (!this.chart) return

      const displayData = this.chartData.slice(0, 10)
      const categories = displayData.map(item => {
        const category = item.category
        return category.includes('(') ? category.split('(')[0].trim() : category
      })
      const values = displayData.map(item => item.count / 1000000)
      const colors = [
        '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
        '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#60acfc'
      ].slice(0, categories.length)

      this.chart.setOption({
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          },
          formatter(params) {
            const param = params[0]
            const originalData = displayData[param.dataIndex]
            if (!originalData) {
              return `${param.name}<br/>数量: ${param.value}`
            }
            return `${originalData.category}<br/>数量: ${originalData.count.toLocaleString()}`
          }
        },
        grid: {
          left: '3%',
          right: '3%',
          top: '15%',
          bottom: '20%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: categories,
          axisTick: {
            alignWithLabel: true,
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#e0e0e0'
            }
          },
          axisLabel: {
            interval: 0,
            rotate: 45,
            fontSize: 11,
            margin: 15,
            color: '#666',
            formatter(value) {
              return value.length > 15 ? `${value.substring(0, 15)}...` : value
            }
          }
        },
        yAxis: {
          type: 'value',
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#e0e0e0'
            }
          },
          name: '数量（百万）',
          nameTextStyle: {
            fontSize: 12,
            padding: [0, 0, 0, 10],
            color: '#666'
          },
          splitLine: {
            lineStyle: {
              type: 'dashed',
              color: '#f0f0f0'
            }
          },
          axisLabel: {
            color: '#666',
            fontSize: 11
          }
        },
        series: [{
          name: '恶意域名类型',
          type: 'bar',
          barWidth: '40%',
          data: values.map((value, index) => ({
            value,
            itemStyle: {
              color: colors[index],
              borderRadius: [2, 2, 0, 0]
            }
          })),
          animationDuration,
          animationEasing: 'cubicOut',
          label: {
            show: true,
            position: 'top',
            formatter(params) {
              const originalData = displayData[params.dataIndex]
              return originalData ? (originalData.count / 1000000).toFixed(1) : params.value.toFixed(1)
            },
            fontSize: 10,
            color: '#333',
            fontWeight: 'bold'
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.3)'
            }
          }
        }]
      })
    }
  }
}
</script>
