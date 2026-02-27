<template>
  <div :class="className" :style="{height:height,width:width}" />
</template>

<script>
import * as echarts from 'echarts'
// ECharts theme
import resize from './mixins/resize.js'
import chartData from '@/data/chart_data.js' // 导入数据

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
    }
  },
  data() {
    return {
      chart: null
    }
  },
  mounted() {
    this.$nextTick(() => {
      this.initChart()
    })
  },
  beforeDestroy() {
    if (!this.chart) {
      return
    }
    this.chart.dispose()
    this.chart = null
  },
  methods: {
    initChart() {
      this.chart = echarts.init(this.$el, 'macarons')

      // 从导入的数据中获取top10Source
      const top10Data = chartData.top10Source || []

      // 提取sources和counts，并按count降序排列（如果数据未排序）
      const sortedData = [...top10Data].sort((a, b) => b.count - a.count)

      // 提取前10个数据
      const displayData = sortedData.slice(0, 10)

      // 将count转换为百万单位，并保留1位小数（根据需要调整）
      // 这里直接使用原始count值，如果数值太大可以考虑转换
      const sources = displayData.map(item => item.source)
      const values = displayData.map(item => item.count / 1000000) // 转换为百万单位

      // 使用 ECharts 默认的鲜艳彩色系列
      const colors = [
        '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
        '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#60acfc'
      ].slice(0, sources.length)

      this.chart.setOption({
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          },
          formatter: function(params) {
            const param = params[0]
            const originalData = displayData[param.dataIndex]
            const originalCount = originalData ? originalData.count.toLocaleString() : param.value
            return `${param.name}<br/>数量: ${originalCount}`
          }
        },
        grid: {
          left: '3%',
          right: '3%',
          top: '15%',
          bottom: '25%', // 增加底部留白以适应长标签
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: sources,
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
            fontSize: 10, // 稍微减小字体大小以适应长标签
            margin: 15,
            color: '#666',
            formatter: function(value) {
              // 如果标签太长，可以截断并添加省略号
              if (value.length > 20) {
                return value.substring(0, 20) + '...'
              }
              return value
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
          name: '数量（百万）', // 修改单位说明
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
            fontSize: 11,
            formatter: function(value) {
              return value
            }
          }
        },
        series: [{
          name: '恶意域名来源',
          type: 'bar',
          barWidth: '40%',
          data: values.map((value, index) => ({
            value: value,
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
            formatter: function(params) {
              const originalData = displayData[params.dataIndex]
              // 显示原始count值（已转换为百万单位并保留1位小数）
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
