// 获得网页组件
var totalFileStatistics = document.getElementById("TotalFileStatistics");
var totalFileStatistics_detail = document.getElementById("TotalFileStatistics_detail");

// 数据预处理
var temp = MyArguments.totalFileNumber.toString();
var totalFileStatistics_detail_str = "";
var totalFileStatistics_str = "";
for(var index = 0;index < temp.length;index++) {
    if(index % 3 == 0 && index !== 0) totalFileStatistics_detail_str = " " + totalFileStatistics_detail_str;
    totalFileStatistics_detail_str = temp.charAt(temp.length - index - 1) + totalFileStatistics_detail_str;
}
if(MyArguments.totalFileNumber > 100000000) totalFileStatistics_str = Math.round(MyArguments.totalFileNumber/100000000) + "亿";
else if(MyArguments.totalFileNumber > 10000) totalFileStatistics_str = Math.round(MyArguments.totalFileNumber/10000) + "万";
else totalFileStatistics_str = MyArguments.totalFileNumber.toString();

// 显示
totalFileStatistics_detail.innerText = "具体数量: " + totalFileStatistics_detail_str + '个';
totalFileStatistics.innerText = totalFileStatistics_str;
