<template>
  <div :class="className" :style="{height:height,width:width}" />
</template>

<script>
import echarts from 'echarts'
require('echarts/theme/macarons') // echarts theme
import resize from './mixins/resize'
// 导入生成的JS数据文件
import chartData from '@/data/chart_data.js'

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
      default: '300px'
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

      this.chart.setOption({
        tooltip: {
          trigger: 'item',
          formatter: '{a} <br/>{b} : {c} ({d}%)'
        },
        legend: {
          left: 'center',
          bottom: '10',
          // 支持滚动，避免平台类别过多时溢出
          type: 'scroll',
          pageButtonPosition: 'end',
          pageIconColor: '#666',
          pageIconInactiveColor: '#ccc',
          pageTextStyle: {
            color: '#999'
          }
        },
        series: [
          {
            name: '恶意文件样本平台Top10',
            type: 'pie',
            roseType: 'radius',
            radius: [15, 95],
            center: ['50%', '38%'],
            // 从JS文件读取platform的Top10+Others数据
            data: chartData.pieTop10Data.platform,
            animationEasing: 'cubicInOut',
            animationDuration: 2600,
            // 平台名称标签优化
            label: {
              formatter: '{b}: {d}%',
              fontSize: 12,
              // 长名称自动截断
              overflow: 'truncate',
              ellipsis: '...',
              width: 100 // 标签最大宽度
            },
            // 鼠标悬停时显示完整名称
            emphasis: {
              label: {
                fontSize: 14,
                fontWeight: 'bold',
                overflow: 'none', // 取消截断
                width: 'auto'
              }
            }
          }
        ]
      })
    }
  }
}
</script>