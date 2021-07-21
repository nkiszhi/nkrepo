var scriptState = document.getElementById("ScriptState");
scriptState.innerHTML = "";
for(var item of MyArguments.scriptState) {
    if(item[1] > 100000000) item[1] = Math.round(item[1]/100000000) + "亿";
    else if(item[1] > 10000) item[1] = Math.round(item[1]/10000) + "万";
    else item[1] = item[1].toString();
    scriptState.innerHTML += '<li><div class="fontInner clearfix" style="height = 20px;"><span>' + item[0] + '</span><span>' + item[1] + '</span></div></li>';
}

$(function () {

    ceshis();
    ceshis1();
    ceshis2();
    ceshis3();
    ceshis4();

    function ceshis() {
        var myChart = echarts.init(document.getElementById('chart1'));
        var legend_data_type = [];
        var series_data_type = [];
        for(var item of MyArguments.fileTypeDistribution) {
            legend_data_type.push(item[0]);
            series_data_type.push({value : item[1], name : item[0]});
        }
        option = {
            /*backgroundColor: '#05163B',*/
            tooltip: {
                trigger: 'axis'
            },
            toolbox: {
                show: true,
                feature: {
                    mark: {
                        show: true
                    },
                    dataView: {
                        show: true,
                        readOnly: false
                    },
                    magicType: {
                        show: true,
                        type: ['line', 'bar']
                    },
                    restore: {
                        show: true
                    },
                    saveAsImage: {
                        show: true
                    }
                }
            },
            grid: {
                top: 'middle',
                left: '3%',
                right: '4%',
                bottom: '3%',
                top: '10%',
                containLabel: true
            },
            legend: {
                data: legend_data_type,
                textStyle: {
                    color: "#fff"
                }

            },
            xAxis: [{
                type: 'category',
                data: legend_data_type,
                axisLabel: {
                    show: true,
                    textStyle: {
                        color: "#ebf8ac" //X轴文字颜色
                    }
                },
                axisLine: {
                    lineStyle: {
                        color: '#01FCE3'
                    }
                },
            }],
            yAxis: [
                {
                    type: 'value',
                    name: '数量',
                    axisLabel: {
                        formatter: '{value} 个',
                        textStyle: {
                            color: "#2EC7C9" //X轴文字颜色
                        }
                    },axisLine: {
                        lineStyle: {
                            color: '#01FCE3'
                        }
                    },
                },
            ],
            series: [
                {
                    name: '数量',
                    type: 'bar',
                    itemStyle: {
                        normal: {
                            barBorderRadius: 5,
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                                offset: 0,
                                color: "#C1B2EA"
                            },
                                {
                                    offset: 1,
                                    color: "#8362C6"
                                }
                            ])
                        }
                    },
                    data: series_data_type,
                },
            ]
        };

        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }

    function ceshis1() {
        var myChart = echarts.init(document.getElementById('chart2'));
        var ydata = []
        var xdata = []
        var color = ["#8d7fec", "#5085f2", "#e75fc3", "#f87be2", "#f2719a", "#fca4bb", "#f59a8f", "#fdb301", "#57e7ec", "#cf9ef1"]
        for(var item of MyArguments.fileSizeDistribution) {
            xdata.push(item[0]);
            ydata.push({value : item[1], name : item[0]});
        }

        option = {
            /*backgroundColor: "rgba(255,255,255,1)",*/
            // color: color,
            legend: {
                orient: "vartical",
                x: "left",
                top: "center",
                left: "53%",
                bottom: "0%",
                data: xdata,
                itemWidth: 8,
                itemHeight: 8,
                textStyle: {
                    color: '#fff'
                },
                /*itemGap: 16,*/
                /*formatter:function(name){
                  var oa = option.series[0].data;
                  var num = oa[0].value + oa[1].value + oa[2].value + oa[3].value+oa[4].value + oa[5].value + oa[6].value + oa[7].value+oa[8].value + oa[9].value ;
                  for(var i = 0; i < option.series[0].data.length; i++){
                      if(name==oa[i].name){
                          return ' '+name + '    |    ' + oa[i].value + '    |    ' + (oa[i].value/num * 100).toFixed(2) + '%';
                      }
                  }
                }*/

            },
            series: [{
                type: 'pie',
                clockwise: false, //饼图的扇区是否是顺时针排布
                minAngle: 2, //最小的扇区角度（0 ~ 360）
                radius: ["20%", "60%"],
                center: ["30%", "45%"],
                avoidLabelOverlap: false,
                itemStyle: { //图形样式
                    normal: {
                        borderColor: '#ffffff',
                        borderWidth: 1,
                    },
                },
                label: {
                    normal: {
                        show: false,
                        position: 'center',
                        rich: {
                            text: {
                                color: "#fff",
                                fontSize: 14,
                                align: 'center',
                                verticalAlign: 'middle',
                                padding: 8
                            },
                            value: {
                                color: "#8693F3",
                                fontSize: 24,
                                align: 'center',
                                verticalAlign: 'middle',
                            },
                        }
                    },
                    emphasis: {
                        show: true,
                        textStyle: {
                            fontSize: 24,
                        }
                    }
                },
                data: ydata
            }]
        };
        myChart.setOption(option);

        setTimeout(function() {
            myChart.on('mouseover', function(params) {
                if (params.name == ydata[0].name) {
                    myChart.dispatchAction({
                        type: 'highlight',
                        seriesIndex: 0,
                        dataIndex: 0
                    });
                } else {
                    myChart.dispatchAction({
                        type: 'downplay',
                        seriesIndex: 0,
                        dataIndex: 0
                    });
                }
            });

            myChart.on('mouseout', function(params) {
                myChart.dispatchAction({
                    type: 'highlight',
                    seriesIndex: 0,
                    dataIndex: 0
                });
            });
            myChart.dispatchAction({
                type: 'highlight',
                seriesIndex: 0,
                dataIndex: 0
            });
        }, 1000);

        // 使用刚指定的配置项和数据显示图表。
        /*myChart.setOption(option);*/
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
    function ceshis2() {
        var myChart = echarts.init(document.getElementById('chart3'));

        var legend_data = [];
        var series_data = [];
        for(var item of MyArguments.familyDistribution) {
            legend_data.push(item[0]);
            series_data.push({value : item[1], name : item[0]});
        }
        option = {
            /*backgroundColor: '#000',*/
            "animation": true,
            "title": {
                /*"text": 24,*/
                "x": "center",
                "y": "center",
                "textStyle": {
                    "color": "#fff",
                    "fontSize": 10,
                    "fontWeight": "normal",
                    "align": "center",
                    "width": "200px"
                },
                "subtextStyle": {
                    "color": "#fff",
                    "fontSize": 10,
                    "fontWeight": "normal",
                    "align": "center"
                }
            },
            "legend": {
                "show" : false,
                "width": "100%",
                "left": "center",
                "textStyle": {
                    "color": "#fff",
                    "fontSize": 10
                },
                "icon": "circle",
                "right": "0",
                "bottom": "0",
                "padding": [70, 20],
                "itemGap": 5,
                "data": legend_data
            },
            "series": [{
                "type": "pie",
                "center": ["50%", "40%"],
                "radius": ["20%", "43%"],
                "color": ["#FEE449", "#00FFFF", "#00FFA8", "#9F17FF", "#FFE400", "#F76F01", "#01A4F7", "#FE2C8A"],
                "startAngle": 135,
                "labelLine": {
                    "normal": {
                        "length": 15
                    }
                },
                "label": {
                    "normal": {
                        "formatter": "{b|{b}:}  {per|{d}%} ",
                        "backgroundColor": "rgba(255, 147, 38, 0)",
                        "borderColor": "transparent",
                        "borderRadius": 4,
                        "rich": {
                            "a": {
                                "color": "#999",
                                "lineHeight": 12,
                                "align": "center"
                            },
                            "hr": {
                                "borderColor": "#aaa",
                                "width": "100%",
                                "borderWidth": 1,
                                "height": 0
                            },
                            "b": {
                                "color": "#b3e5ff",
                                "fontSize": 14,
                                "lineHeight": 33
                            },
                            "c": {
                                "fontSize": 10,
                                "color": "#eee"
                            },
                            "per": {
                                "color": "#FDF44E",
                                "fontSize": 10,
                                "padding": [5, 8],
                                "borderRadius": 2
                            }
                        },
                        "textStyle": {
                            "color": "#fff",
                            "fontSize": 16
                        }
                    }
                },
                "emphasis": {
                    "label": {
                        "show": true,
                        "formatter": "{b|{b}:}  {per|{d}%}  ",
                        "backgroundColor": "rgba(255, 147, 38, 0)",
                        "borderColor": "transparent",
                        "borderRadius": 4,
                        "rich": {
                            "a": {
                                "color": "#999",
                                "lineHeight": 22,
                                "align": "center"
                            },
                            "hr": {
                                "borderColor": "#aaa",
                                "width": "100%",
                                "borderWidth": 1,
                                "height": 0
                            },
                            "b": {
                                "color": "#fff",
                                "fontSize": 10,
                                "lineHeight": 33
                            },
                            "c": {
                                "fontSize": 10,
                                "color": "#eee"
                            },
                            "per": {
                                "color": "#FDF44E",
                                "fontSize": 10,
                                "padding": [5, 6],
                                "borderRadius": 2
                            }
                        }
                    }
                },
                "data": series_data
            }, {
                "type": "pie",
                "center": ["50%", "40%"],
                "radius": ["15%", "14%"],
                "label": {
                    "show": false
                },
                "data": [{
                    "value": 78,
                    "name": "实例1",
                    "itemStyle": {
                        "normal": {
                            "color": {
                                "x": 0,
                                "y": 0,
                                "x2": 1,
                                "y2": 0,
                                "type": "linear",
                                "global": false,
                                "colorStops": [{
                                    "offset": 0,
                                    "color": "#9F17FF"
                                }, {
                                    "offset": 0.2,
                                    "color": "#01A4F7"
                                }, {
                                    "offset": 0.5,
                                    "color": "#FE2C8A"
                                }, {
                                    "offset": 0.8,
                                    "color": "#FEE449"
                                }, {
                                    "offset": 1,
                                    "color": "#00FFA8"
                                }]
                            }
                        }
                    }
                }]
            }]
        }

        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
    function ceshis3() {
        var myChart = echarts.init(document.getElementById('chart4'));

        var colors = ['rgb(46, 199, 201)', 'rgb(255, 185, 128)'];
        var legend_data = [];
        var series_data = [];
        var add_data = [];
        var flag = -1;
        for(var item of MyArguments.totalSampleData) {
            if(flag == -1){
                flag = item[1]
                continue
            }
            legend_data.push(item[0]);
            series_data.push(item[1]);
            add_data.push(item[1] - flag);
            flag = item[1];
        }

        option = {
            color: colors,

            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross'
                },
            },
            grid: {
                right: '20%'
            },
            toolbox: {
                feature: {
                    dataView: {
                        show: true,
                        readOnly: false
                    },
                    restore: {
                        show: true
                    },
                    saveAsImage: {
                        show: true
                    }
                }
            },
            legend: {
                textStyle: {
                    color: '#fff'
                },
                data: ['新增数量', '总数量']
            },
            // 缩放组件
            /*dataZoom: {
                type: 'slider'
            },*/
            xAxis: [{
                type: 'category',
                axisTick: {
                    alignWithLabel: true
                },
                axisLabel: {
                    formatter: '{value} ',
                    textStyle: {
                        color: "#ffffff" //X轴文字颜色
                    }
                },
                data: legend_data,
            }],
            yAxis: [{
                type: 'value',
                name: '新增数量',
                min: 0,
                position: 'right',
                axisLine: {
                    lineStyle: {
                        color: colors[0]
                    }
                },
                axisLabel: {
                    formatter: '{value} 个'
                }
            },
                {
                    type: 'value',
                    name: '总数量',
                    min: 0,
                    position: 'left',
                    axisLine: {
                        lineStyle: {
                            color: colors[1]
                        }
                    },
                    axisLabel: {
                        formatter: '{value} 个'
                    }
                }
            ],
            series: [{
                name: '新增数量',
                type: 'bar',
                data: add_data,
                itemStyle: {
                    normal: {
                        barBorderRadius: 2
                    }
                }
            },
                {
                    name: '总数量',
                    type: 'line',
                    yAxisIndex: 1,
                    data: series_data,
                }
            ]
        };
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }
    function ceshis4() {
        var myChart = echarts.init(document.getElementById('chart5'));

        var labelimg = "/asset/get/s/data-1575019476644-Rak5eXt1.png";
        var xAxis_data_time = [];
        var series_data_time = [];
        for(var item of MyArguments.fileTimeDistribution) {
            xAxis_data_time.push(item[0]);
            series_data_time.push({
                "name": "",
                "value": Math.round(item[1] / 10000),
                "label": {
                    "show": true,
                    "position": "top",
                    "color" : "#FFFFFF",
                    "rich": {
                        "a": {
                            "fontSize": 18,
                            "color": "#534EE1",
                            "align": "center",
                            "height": 40
                        },
                        "c": {
                            "fontSize": 18,
                            "color": "#fff",
                            "padding": [
                                -2,
                                0,
                                2,
                                0
                            ],
                            "backgroundColor": {
                                "image": labelimg
                            },
                            "align": "center",
                            "verticalAlign": "bottom",
                            "height": 50,
                            "lineHeight": 40,
                            "width": 100
                        }
                    }
                },
                "itemStyle": {
                    "normal": {
                        "color": {
                            "type": "linear",
                            "x": 0,
                            "y": 0,
                            "x2": 0,
                            "y2": 1,
                            "colorStops": [{
                                "offset": 0,
                                "color": "rgba(83,78,225,1)"
                            },
                                {
                                    "offset": 1,
                                    "color": "rgba(83,78,225,0)"
                                }
                            ],
                            "global": false
                        }
                    }
                }
            });
        }
        option = {
            /*backgroundColor: "#0E233E",*/
            "grid": {
                "left": "6%",
                "top": "10%",
                "right": "3%",
                "bottom": "15%"
            },
            "legend": {
                "data": xAxis_data_time,
                "top": "92%",
                "icon": "circle",
                "textStyle": {
                    "color": "#0DCAD2"
                }
            },
            "color": [
                "#534EE1",
                "#ECD64F",
                "#00E4F0",
                "#44D16D",
                "#124E91",
                "#BDC414",
                "#C8CCA5"
            ],
            "tooltip": {
                "position": "top",
            },
            "xAxis": {
                "type": "category",
                "data": xAxis_data_time,
                "axisLabel": {
                    "show": true,
                    "color": "#999999",
                    "fontSize": 10
                },
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {
                    "show": false,
                    "color": "#999999",
                    "fontSize": 16
                },
                "axisTick": {
                    "show": false
                },
                "axisLine": {
                    "show": false
                },
                "splitLine": {
                    "show": false
                }
            },
            "series": [
                {
                    "type": "pictorialBar",
                    "name": "数量",
                    "data": series_data_time,
                    "stack": "a",
                    "symbol": "path://M0,10 L10,10 C5.5,10 5.5,5 5,0 C4.5,5 4.5,10 0,10 z"
                }
            ]
        }
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }

    function ceshis5() {
        var myChart = echarts.init(document.getElementById('chart_1'));


        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
        window.addEventListener("resize",function(){
            myChart.resize();
        });
    }


});