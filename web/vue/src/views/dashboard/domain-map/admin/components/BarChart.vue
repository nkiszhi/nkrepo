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

      // 从导入的数据中获取top10Category
      const top10Category = chartData.top10Category || []

      // 提取categories和counts
      const displayData = top10Category.slice(0, 10) // 取前10个数据

      // 处理category名称，如果太长可以简化
      const categories = displayData.map(item => {
        // 简化category名称显示
        const category = item.category
        // 如果包含括号，可以提取括号前的内容
        if (category.includes('(')) {
          return category.split('(')[0].trim()
        }
        return category
      })

      // 将count转换为百万单位，保留1位小数
      const values = displayData.map(item => item.count / 1000000)

      // 使用鲜艳的彩色系列
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
          formatter: function(params) {
            const param = params[0]
            const originalData = displayData[param.dataIndex]
            if (originalData) {
              const originalCount = originalData.count.toLocaleString()
              const originalCategory = originalData.category
              return `${originalCategory}<br/>数量: ${originalCount}`
            }
            return `${param.name}<br/>数量: ${param.value}`
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
            formatter: function(value) {
              // 如果标签太长，可以截断显示
              if (value.length > 15) {
                return value.substring(0, 15) + '...'
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
            fontSize: 11,
            formatter: function(value) {
              return value
            }
          }
        },
        series: [{
          name: '恶意域名类型',
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
              if (originalData) {
                // 显示百万单位的数值，保留1位小数
                return (originalData.count / 1000000).toFixed(1)
              }
              return params.value.toFixed(1)
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
