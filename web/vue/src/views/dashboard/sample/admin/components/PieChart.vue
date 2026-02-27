<template>
  <div :class="className" :style="{height:height,width:width}" />
</template>

<script>
import * as echarts from 'echarts'
// ECharts theme // echarts theme
import resize from './mixins/resize.js'
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
          // 增加滚动功能，避免标签过多溢出
          type: 'scroll',
          pageButtonPosition: 'end',
          pageIconColor: '#666',
          pageIconInactiveColor: '#ccc'
        },
        series: [
          {
            name: '恶意文件样本平台Top10',
            type: 'pie',
            roseType: 'radius',
            radius: [15, 95],
            center: ['50%', '42%'],
            // 从JS文件读取platform的Top10+Others数据
            data: chartData.pieTop10Data.platform,
            animationEasing: 'cubicInOut',
            animationDuration: 2600,
            // 优化标签显示
            label: {
              formatter: '{b}: {d}%',
              fontSize: 12,
              overflow: 'truncate', // 标签过长时截断
              ellipsis: '...'
            },
            // 鼠标悬停时放大标签
            emphasis: {
              label: {
                fontSize: 14,
                fontWeight: 'bold',
                overflow: 'none' // 悬停时不截断
              }
            }
          }
        ]
      })
    }
  }
}
</script>
