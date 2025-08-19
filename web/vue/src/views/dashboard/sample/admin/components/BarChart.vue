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
          // 处理标签过多的情况，超出时自动滚动
          type: 'scroll',
          pageButtonPosition: 'end',
          pageIconColor: '#aaa',
          pageIconInactiveColor: '#ccc'
        },
        series: [
          {
            name: '恶意文件样本家族Top10',
            type: 'pie',
            roseType: 'radius',
            radius: [15, 95],
            center: ['50%', '38%'],
            // 使用从JS文件中读取的家族Top10+Others数据
            data: chartData.pieTop10Data.family,
            animationEasing: 'cubicInOut',
            animationDuration: 2600,
            // 标签显示优化
            label: {
              formatter: '{b}: {d}%',
              fontSize: 12
            },
            // 鼠标悬停时的标签样式
            emphasis: {
              label: {
                fontSize: 14,
                fontWeight: 'bold'
              }
            }
          }
        ]
      })
    }
  }
}
</script>
