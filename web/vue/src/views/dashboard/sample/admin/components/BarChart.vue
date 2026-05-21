<template>
  <div :class="className" :style="{height:height,width:width}" />
</template>

<script>
import * as echarts from 'echarts'
// ECharts theme // echarts theme
import resize from './mixins/resize.js'

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
    this.$nextTick(() => {
      this.initChart()
    })
  },
  beforeUnmount() {
    if (!this.chart) {
      return
    }
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
            data: this.chartData,
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
