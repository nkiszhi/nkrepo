<template>
  <div :class="className" :style="{height:height,width:width}" />
</template>

<script>
import echarts from 'echarts'
require('echarts/theme/macarons') // echarts theme
import resize from './mixins/resize'

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
          bottom: '10'
        },
        series: [
          {
            name: '恶意文件样本类型Top10',
            type: 'pie',
            roseType: 'radius',
            radius: [15, 95],
            center: ['50%', '38%'],
            data: [
              { value: 254.3, name: 'Trojan' },
              { value: 65.2, name: 'AdWare' },
              { value: 58.8, name: 'Trojan-Dropper' },
              { value: 53.8, name: 'Trojan-Downloader' },
              { value: 22.0, name: 'Downloader' },
              { value: 19.5, name: 'Hoax' },
              { value: 18.0, name: 'DangerousObject' },
              { value: 17.8, name: 'Backdoor' },
              { value: 15.8, name: 'Virus' },
              { value: 12.9, name: 'Worm' },
              { value: 61.9, name: 'Others' }
            ],
            animationEasing: 'cubicInOut',
            animationDuration: 2600
          }
        ]
      })
    }
  }
}
</script>
