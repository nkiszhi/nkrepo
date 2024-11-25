<template>  
  <main>
    <div class="text-center">  
      <h2 class="text-primary">样本SHA256查询</h2>  
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>  
    </div>  
  
    <div class="input-container"> 
      <input type="text" class="input-text" placeholder="请输入一个病毒SHA256进行查询" v-model="tableName" name="category" />  
      <button type="button" class="btn btn-outline-primary" @click="searchVirus" name="detect-category">  
        <!-- 嵌入搜索图标 -->  
        <svg xmlns="http://www.w3.org/2000/svg" width="35px" height="35px" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">  
          <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"></path>  
        </svg>   
      </button> 
   </div>
   <div class="chart-wrapper" style="width:100%; height:200%;background:#fff;padding:16px 16px 0;margin-bottom:32px;">
           <line-chart :chart-data="lineChartData" />
    </div>



   
  <div>  
<div v-if="SearchResult === '查询成功'" class="result-success" style="text-align: center;">  
  <p>{{ SearchResult }}</p>  
</div>  
<div v-if="SearchResult !== '查询成功' && SearchResult !== null && SearchResult !== ' '" class="result-failed" style="text-align: center;">  
  <p>{{ SearchResult }}</p>  
</div>
  
    <div v-if="queryResult && typeof queryResult === 'object' && queryResult.query_sha256 && Object.keys(queryResult.query_sha256).length > 0">  
      <table class="file-info-table" >  
        <tr>  
          <th>查询结果</th> 
          <th> </th> 
        </tr>  
        <tr v-for="(value, key) in queryResult.query_sha256" :key="key" v-if="value !== 'nan' && value !== 'nan\r'">  
          <td>{{ key.replace('_', ' ') }}：</td>  
          <td >{{ value }}</td>  
        </tr> 
        <tr>  
          <td colspan="2" class="download-btn-container">  
            <button class="download-btn" @click="downloadFile(queryResult.query_sha256.SHA256)">下载文件</button>  
          </td>  
        </tr>  
      </table>
    </div> 
</div>
      
  </main> 
</template> 

<script>  
import axios from 'axios';  
import LineChart from '../dashboard/sample/admin/components/LineChart'
const lineChartData = {
  total_amount: {
    date_data: ["2012年","2013年","2014年","2015年","2016年","2017年","2018年","2019年","2020年","2021年","2022年","2023年","2024年"],
    amount_data: [353.9,1140.3,419.4,445.7,412.9,209.7,255.6,176.9,190.1,13.1,334.2,203.2,13.1 ]
  },

}
export default {  
  data() {  
    return {  
      tableName: '',  
      queryResult: null,  
      SearchResult: null, // 初始化 SearchResult  
      lineChartData: lineChartData.total_amount
    };  
  },   
  components: {
   LineChart,
  }, 
  methods: {  
    searchVirus() {  
      if (!this.tableName || this.tableName.length !== 64) {  
        this.SearchResult = '请输入正确的SHA256';  
        return;  
      }  
      axios.post('http://10.134.2.27:5000/query_sha256', { tableName: this.tableName })  
        .then(response => {    
          // 确保 response.data 是一个对象  
          if (typeof response.data === 'string') {  
            try {  
              this.queryResult = JSON.parse(response.data);
              this.SearchResult = '查询成功';  
            } catch (e) {  
              console.error('Failed to parse JSON:', e);  
              this.queryResult = null; // 或者设置一个错误状态  
            }  
          } else {  
            this.queryResult = response.data;
            this.SearchResult = '查询成功';

          }  
        })
        .catch(error => {  
          console.error('Error fetching data:', error);  
          this.queryResult = null;  
          this.SearchResult = '未查询到此样本';  
        });
      },
   downloadFile(sha256) {  
    const downloadUrl = `http://10.134.2.27:5000/download_sha256/${sha256}`;  
  
    // 创建一个新的a标签  
    const link = document.createElement('a');  
    link.href = downloadUrl;  
    link.setAttribute('download', ''); // 触发下载行为，可以根据需要设置下载的文件名  
    link.style.display = 'none'; // 隐藏a标签
    this.$forceUpdate();  
  
    // 触发点击事件  
    document.body.appendChild(link);  
    link.click();  
    document.body.removeChild(link); // 清理创建的a标签
    this.$forceUpdate();   
  }  
}   
};  
</script>
<style scoped>  
.text-center {  
  /* 文本居中的样式 */  
  text-align: center;  
} 
.input-container {  
  /* 使用Flexbox布局使子元素居中 */  
  display: flex;  
  justify-content: center; /* 水平居中 */  
  align-items: center; /* 垂直居中（如果.input-container有高度） */  
  height: 100px; /* 为垂直居中设置一个高度，或者根据需要调整 */  
  margin-top: 10px; /* 与上方的元素间隔一些距离 */  
}  
  
.input-text {  
  /* 输入框的样式 */  
  width: 800px; /* 设置输入框的宽度 */  
  height: 50px; /* 设置输入框的高度 */  
  padding: 5px; /* 设置内边距 */  
  border-radius: 5px; /* 设置边框圆角 */  
  border: 1px solid #ccc; /* 设置边框 */  
}  
.result-content {  
  display: flex;  
  justify-content: center;
  align-items: center; /* 垂直居中 */  
}  
.result-inner {  
  /* 使用Flexbox使图标和文本居中 */  
  display: flex;  
  align-items: center; /* 垂直居中 */  
  justify-content: center;
}  
  
.result-icon {  
  /* 假设你的图标是SVG，并且你想设置它的大小为32x32像素 */  
  width: 40px;  
  height: 40px;  
  margin-right: 20px; /* 图标和文本之间的间距 */  
  margin-top:5px;
}  

.result-status {  
  /* 状态文本的样式 */  
  font-size: 30px;  
  color: #333;  
  font-weight: bold;  
  text-justify:center;
}  
  
.result-reason {  
  /* 原因文本的样式 */  
  font-size: 20px;  
  color: #666;  
  margin-top: 20px; /* 和上面的内容保持一定间距 */  
  text-align: center; /* 如果需要的话，可以将原因文本左对齐 */  
}  
 
/* 通用表格样式 */  
.file-info-table {  
  width: 55%; /* 调整为55%或您需要的任何宽度 */  
  margin: 30px auto; /* 顶部外边距和自动左右外边距 */  
  border: 1px solid #ccc;  
  border-collapse: collapse;  
}  
  
.file-info-table th,  
.file-info-table td {  
  padding: 8px;  
  text-align: left;  
  border-bottom: 1px solid #ddd;  
}  
  
.file-info-table tr:hover {  
  background-color: #f5f5f5;  
}  
  
/* 结果成功和失败的样式 */  
.result-success,  
.result-failed {  
  margin: 20px auto;  
  padding: 20px;  
  width: 60%;  
  max-width: 850px;  
  border-radius: 10px;  
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  
  text-align: center;  
  font-size: 16px;  
  font-weight: bold;  
}  
  
.result-success {  
  background-color: #d4edda;  
  border: 2px solid #c3e6cb;  
  color: #155724;  
}  
  
.result-failed {  
  background-color: #f8d7da;  
  border: 2px solid #f5c6cb;  
  color: #721c24;  
}  
  
/* 下载按钮容器和按钮样式 */  
.download-btn-container {  
  text-align: right; /* 如果需要按钮右对齐 */  
}  
  
.download-btn {  
  display: inline-block; /* 确保按钮宽度设置有效 */  
  width: 100%; /* 宽度设置为100%，但请注意这会影响包含它的容器 */  
  background-color: #007bff;  
  color: white;  
  border: none;  
  border-radius: 5px;  
  padding: 10px 20px;  
  font-size: 16px;  
  font-weight: bold;  
  transition: background-color 0.3s;  
  cursor: pointer;  
  /* 移除margin-left和margin-right，因为width: 100%已经使其占满容器 */  
}  
  
.download-btn:hover {  
  background-color: #0056b3;  
}
</style>

