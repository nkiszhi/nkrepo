<template>  
  <main>
    <div class="text-center">  
      <h2 class="text-primary">样本类型查询</h2>  
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>  
    </div>  
  
    <div class="input-container"> 
      <input type="text" class="input-text" placeholder="请输入一个病毒类型进行查询" v-model="tableName" name="category" />  
      <button type="button" class="btn btn-outline-primary" @click="searchVirus" name="detect-category">  
        <!-- 嵌入搜索图标 -->  
        <svg xmlns="http://www.w3.org/2000/svg" width="35px" height="35px" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">  
          <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"></path>  
        </svg>   
      </button>  
    </div>  
    
    <div class="chart-wrapper" style="width:100%; height:200%;">
           <pie1-chart />
    </div>

<div v-if="isLoading" class="search-status">  
      <svg-icon icon-class="file_search" class="result-icon" />  
      <span class="icon file-search"></span> 样本搜索中，请稍等...  
    </div> 
<div v-else-if="searchResults.length > 0" class="search-results">  
<h2>符合条件的样本如下：</h2>
  <ul class="flex-container"> 
    <li v-for="(sha256, index) in searchResults":key="sha256" class="table-row" @mouseover="hoveringOver = sha256"  @mouseleave="hoveringOver = null" @click="toggleDetails(sha256)">  
      <div class="table-row-inner flex-row" style="width: 100%; margin: 0 auto;">  
        <div class="table-cell sha256-cell">  
          <span class="label">样本{{ index + 1 }}  {{ sha256 }}</span>
        </div>  
      </div>  
      
      <div v-if="showDetails[sha256]" class="table-cell details-cell" style="width:90%; margin: 0 auto;">  
        <p>MD5: {{ details[sha256]['MD5'] }}</p>  
        <p>SHA256: {{ details[sha256]['SHA256'] }}</p>  
        <p>类型: {{ details[sha256]['类型'] }}</p>  
        <p>平台: {{ details[sha256]['平台'] }}</p>  
        <p>家族: {{ details[sha256]['家族'] }}</p>
        <p v-if="details[sha256]['文件拓展名'] !== 'nan'">文件拓展名: {{ details[sha256]['文件拓展名'] }}</p>
        <p v-if="details[sha256]['脱壳'] !== 'nan'">脱壳: {{ details[sha256]['脱壳'] }}</p>
        <p v-if="details[sha256]['SSDEEP'] !== 'nan'">SSDEEP: {{ details[sha256]['SSDEEP'] }}</p> 
        <button class="action-button" @click="downloadFile(sha256)">下载</button>
      </div>  
    </li>  
  </ul>  
</div>
      
  </main> 
</template> 

<script>  
import axios from 'axios';  
import Pie1Chart from '../dashboard/sample/admin/components/Pie1Chart';  
export default {  
  data() {  
    return {  
      tableName: '',  
      sha256s: [],
      details: {},
      hoveringOver: null, // 控制悬浮框的显示 
      showDetails:{},
      searchQuery: '',  
      isLoading: false,  
      searchResults: [], 
    };  
  },
  components: {
    Pie1Chart,
  },  
  methods: {  
    searchVirus() {  
      if (!this.tableName) return;
      this.isLoading = true;  
      this.searchResults = [];
      
      axios.post('http://10.134.2.27:5005/query_category', { tableName: this.tableName })  
        .then(response => {  
          this.searchResults = response.data.sha256s; // 假设后端返回的数据中包含一个名为sha256s的数组  
          this.isLoading = false;
        })  
    .catch(error => {  
      this.isLoading = false;
      if (error.response) {  
        // 请求已发出，但服务器响应的状态码不在 2xx 范围内  
        if (error.response.status === 500) {  
          // 显示服务器内部错误的自定义消息  
          alert('样本库未查询到');  
          // 如果需要，你还可以显示后端返回的具体错误信息  
          // alert('服务器内部错误: ' + error.response.data.error);  
        } else {  
          // 处理其他HTTP错误  
          alert('HTTP错误: ' + error.response.status);  
        }  
      } else if (error.request) {  
        // 请求已发出，但没有收到响应  
        // 这通常发生在请求超时或网络错误时  
        alert('请求已发出但未收到响应');  
      } else {  
        // 一些其他错误发生了  
        alert('发生了未知错误，请检查你的网络连接');  
      }  
      console.error(error);  
    }); 
    },  


    toggleDetails(sha256) {  
     this.$set(this.showDetails, sha256, !this.showDetails[sha256]); 
     axios.get(`http://10.134.2.27:5005/detail_category/${sha256}`) 
         .then(response => {  
           const result = response.data.query_result; 
           this.details[sha256] = result;
           this.$forceUpdate();
           this.isLoading = false;
         		console.log(this.details[sha256])
          })
        .catch(error => {  
          console.error(error);   
        });
    },  
    
    
    downloadFile(sha256) {  
      // 构造下载链接  
      const downloadUrl = `http://10.134.2.27:5005/download_category/${sha256}`;  
    
    // 创建一个新的a标签  
      const link = document.createElement('a');  
      link.href = downloadUrl;  
      link.setAttribute('download', ''); // 触发下载行为  
      this.$forceUpdate();
    
    // 触发点击事件  
      document.body.appendChild(link);  
      link.click();  
      document.body.removeChild(link); // 清理创建的a标签
      this.$forceUpdate();  
   },  

  },    
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

.chart-wrapper {
    background: #fff;
    padding: 20px; /* 如果需要的话 */  
    box-sizing: border-box;
  }

.flex-container {  
  display: flex;  
  flex-direction: column; /* 假设你想垂直排列列表项 */  
  align-items: center; /* 垂直居中（如果需要水平居中，请调整flex-direction） */  
  list-style: none; /* 移除列表项前的默认标记 */  
  padding: 0; /* 移除默认的padding */  
  margin: 0; /* 移除默认的margin */  
}  
  
/* 列表项样式 */  
.table-row {  
  border: 1px solid #ccc; /* 设置边框 */  
  border-radius: 0px; /* 可选：设置边框圆角 */  
  margin-bottom: 0px; /* 列表项之间的间隔 */  
  width:50%; /* 宽度根据需要调整，这里设为100%以填充容器宽度 */  
  position: relative; /* 为可能的子元素定位做准备 */ 
  justify-content: center;  
}  
  
/* 内部容器样式，用于控制内容布局和样式 */  
.table-row-inner {  
  display: flex;  
  justify-content: center; /* 水平居中内容 */  
  align-items: left; /* 垂直居中内容（如果需要） */  
  padding: 10px; /* 设置内边距 */  
}  
  
/* 单元格样式 */  
.table-cell, .sha256-cell {  
  /* 这里可能不需要特别的样式，除非你想进一步定制 */  
  /* 例如，你可以设置文本对齐方式、字体大小等 */  
  text-align: left; /* 如果需要左对齐文本 */  
  font-size: 16px; /* 字体大小 */  
}  
  
/* 悬停样式（如果需要） */  
.table-row:hover {  
  background-color: #f0f0f0; /* 鼠标悬停时的背景色 */  
}
/* 样式化按钮 */  
.action-button {  
  margin-left: 10px;
  padding: 5px 10px; /* 添加内边距 */  
  background-color: #4CAF50; /* 添加背景色 */  
  color: white; /* 设置文本颜色 */  
  border: none; /* 移除边框 */  
  border-radius: 5px; /* 添加圆角 */  
  cursor: pointer; /* 添加鼠标悬停时的样式 */  
  transition: background-color 0.3s ease; /* 添加过渡效果 */  
} 
.action-button:hover {  
  background-color: #45a049; /* 鼠标悬停时的背景色 */  
} 
.details-cell {  
  padding: 10px 0; /* 上下内边距，与header-cell保持一致 */  
  /* 详细信息单元格的其他样式 */  
}
.search-status,.search-results h2 {  
  font-family: 'Arial', sans-serif; /* 使用Arial字体，或者你可以换成任何你喜欢的字体 */  
  /* 你可以添加更多的字体样式属性，如字体大小、颜色等 */  
  font-size: 24px; /* 不同的字体大小 */  
  color: #007BFF; /* 字体颜色示例 */ 
  text-align: center; 
}  
  

  
</style>

