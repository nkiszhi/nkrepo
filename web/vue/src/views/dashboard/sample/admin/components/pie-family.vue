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
      this.$_resizeHandler && this.$_resizeHandler()
      setTimeout(() => {
        this.$_resizeHandler && this.$_resizeHandler()
      }, 120)
    },
    setOptions() {
      if (!this.chart) return
      const chartWidth = this.$el?.clientWidth || window.innerWidth
      const isNarrow = chartWidth < 520

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
            radius: isNarrow ? ['12%', '48%'] : ['10%', '68%'],
            center: isNarrow ? ['50%', '42%'] : ['50%', '38%'],
            data: this.chartData,
            animationEasing: 'cubicInOut',
            animationDuration: 2600,
            // 标签显示优化
            label: {
              show: !isNarrow,
              formatter: '{b}: {d}%',
              fontSize: isNarrow ? 10 : 12,
              overflow: 'truncate',
              ellipsis: '...'
            },
            labelLine: {
              show: !isNarrow
            },
            // 鼠标悬停时的标签样式
            emphasis: {
              label: {
                show: true,
                fontSize: isNarrow ? 11 : 14,
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

<style scoped>
.chart {
  display: block;
  max-width: 100%;
  min-width: 0;
}
</style>
