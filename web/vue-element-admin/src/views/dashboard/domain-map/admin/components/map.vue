<template>
  <div ref="chart" style="width: 100%; height: 600px;"></div>
</template>

<script>
import * as echarts from 'echarts';
import 'echarts/map/js/world'; // 引入世界地图数据

export default {
  name: 'MapWithLines',
  mounted() {
    this.initChart();
  },
  methods: {
    initChart() {
      const chartDom = this.$refs.chart;
      const myChart = echarts.init(chartDom);

      // 城市列表及坐标
      const cities = [
        { name: '北京', coord: [116.40, 39.90] },
        { name: '上海', coord: [121.47, 31.23] },
        { name: '广州', coord: [113.23, 23.16] },
        { name: '深圳', coord: [114.05, 22.54] },
        { name: '成都', coord: [104.06, 30.67] },
        // ... 添加更多城市
      ];

      // 天津的坐标
      const tianjinCoord = [117.20, 39.08];

      // 构建线条数据
      const linesData = cities.map(city => ({
        coords: [city.coord, tianjinCoord],
      }));

      // 天津标注数据
      const tianjinLabelData = [
        {
          name: '天津',
          value: tianjinCoord,
        },
      ];

      const option = {
        tooltip: {
          trigger: 'item',
        },
        geo: {
          map: 'world',
          roam: true,
          label: {
            emphasis: {
              show: false,
            },
          },
          itemStyle: {
            normal: {
              areaColor: '#099ff',
              borderColor: '#111',
            },
            emphasis: {
              areaColor: '#2a333d',
            },
          },
        },
        series: [
          {
            name: '汇聚线',
            type: 'lines',
            coordinateSystem: 'geo',
            zlevel: 2,
            effect: {
              show: true,
              period: 6,
              trailLength: 0.7,
              color: '#fff',
              symbolSize: 3,
            },
            lineStyle: {
              normal: {
                color: '#a6c84c',
                width: 0,
                curveness: 0.2,
              },
            },
            data: linesData,
          },
          {
            name: '城市标注',
            type: 'scatter',
            coordinateSystem: 'geo',
            data: tianjinLabelData,
            symbolSize: 10,
            label: {
              normal: {
                formatter: '{b}',
                position: 'right',
                show: true,
              },
              emphasis: {
                show: true,
              },
            },
            itemStyle: {
              normal: {
                color: '#ffa500',
              },
            },
          },
        ],
      };

      myChart.setOption(option);
    },
  },
};
</script>

<style scoped>
/* 你可以根据需要添加样式 */
</style>