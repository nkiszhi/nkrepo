<template>  
  <main>
    <div class="text-center">  
      <h2 class="text-primary">样本家族查询</h2>  
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>  
    </div>  
  
    <div class="input-container"> 
      <input type="text" class="input-text" placeholder="请输入一个病毒家族进行查询" v-model="tableName" name="family" />  
      <button type="button" class="btn btn-outline-primary" @click="searchVirus" name="detect-family">  
        <!-- 嵌入搜索图标 -->  
        <svg xmlns="http://www.w3.org/2000/svg" width="35px" height="35px" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">  
          <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"></path>  
        </svg>   
      </button>  
    </div>  
    
    <div class="chart-wrapper" style="width:100%; height:200%;">
           <bar-chart />
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
import barChart from '../dashboard/sample/admin/components/BarChart';  
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
      apiBaseUrl: 'http://xxxx242:5005' // 默认API地址
    };  
  },
  
  created() {
    // 组件初始化时加载配置
    this.loadConfig();
  },
  
  components: {
    barChart,
  },  
  methods: {  
    // 从配置文件加载API地址
    async loadConfig() {
      try {
        // 与其他页面保持一致的配置文件路径
        const response = await axios.get('/config.ini', {
          responseType: 'text'
        });
        
        // 解析INI格式配置
        const configContent = response.data;
        const lines = configContent.split('\n');
        let inApiSection = false;
        
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine === '[api]') {
            inApiSection = true;
            continue;
          }
          
          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=');
            if (parts.length >= 2) {
              this.apiBaseUrl = parts[1].trim();
              console.log('从配置文件加载API地址:', this.apiBaseUrl);
              break;
            }
          }
          
          if (inApiSection && trimmedLine.startsWith('[')) {
            break;
          }
        }
      } catch (error) {
        console.warn('加载配置文件失败，使用默认API地址:', error.message);
      }
    },

    searchVirus() {  
      if (!this.tableName) return;
      this.isLoading = true;  
      this.searchResults = [];
      
      // 使用动态API地址
      axios.post(`${this.apiBaseUrl}/query_family`, { tableName: this.tableName })  
        .then(response => {  
          this.searchResults = response.data.sha256s;  
          this.isLoading = false;
        })  
    .catch(error => {  
      this.isLoading = false;
      if (error.response) {  
        if (error.response.status === 500) {  
          alert('样本库未查询到');  
        } else {  
          alert('HTTP错误: ' + error.response.status);  
        }  
      } else if (error.request) {  
        alert('请求已发出但未收到响应');  
      } else {  
        alert('发生了未知错误，请检查你的网络连接');  
      }  
      console.error(error);  
    }); 
    },  


    toggleDetails(sha256) {  
     this.$set(this.showDetails, sha256, !this.showDetails[sha256]); 
     
     // 使用动态API地址
     axios.get(`${this.apiBaseUrl}/detail_family/${sha256}`) 
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
      // 构造动态下载链接
      const downloadUrl = `${this.apiBaseUrl}/download_family/${sha256}`;  
    
      const link = document.createElement('a');  
      link.href = downloadUrl;  
      link.setAttribute('download', '');  
      this.$forceUpdate();
    
      document.body.appendChild(link);  
      link.click();  
      document.body.removeChild(link); 
      this.$forceUpdate();  
   },  

  },    
};  
</script>
<style scoped>  
/* 样式部分保持不变 */
.text-center {  
  text-align: center;  
} 
.input-container {  
  display: flex;  
  justify-content: center;  
  align-items: center;  
  height: 100px;  
  margin-top: 10px;  
}  
  
.input-text {  
  width: 800px;  
  height: 50px;  
  padding: 5px;  
  border-radius: 5px;  
  border: 1px solid #ccc;  
}  

.chart-wrapper {
    background: #fff;
    padding: 20px;  
    box-sizing: border-box;
  }

.flex-container {  
  display: flex;  
  flex-direction: column;  
  align-items: center;  
  list-style: none;  
  padding: 0;  
  margin: 0;  
}  
  
.table-row {  
  border: 1px solid #ccc;  
  border-radius: 0px;  
  margin-bottom: 0px;  
  width:50%;  
  position: relative;  
  justify-content: center;  
}  
  
.table-row-inner {  
  display: flex;  
  justify-content: center;  
  align-items: left;  
  padding: 10px;  
}  
  
.table-cell, .sha256-cell {  
  text-align: left;  
  font-size: 16px;  
}  
  
.table-row:hover {  
  background-color: #f0f0f0;  
}

.action-button {  
  margin-left: 10px;
  padding: 5px 10px;  
  background-color: #4CAF50;  
  color: white;  
  border: none;  
  border-radius: 5px;  
  cursor: pointer;  
  transition: background-color 0.3s ease;  
} 

.action-button:hover {  
  background-color: #45a049;  
} 

.details-cell {  
  padding: 10px 0;  
}

.search-status,.search-results h2 {  
  font-family: 'Arial', sans-serif;  
  font-size: 24px;  
  color: #007BFF;  
  text-align: center; 
}  
</style>