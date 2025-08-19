<template>  
  <main>
    <div class="text-center">  
      <h2 class="text-primary">基于可信度评估的多模型恶意域名检测</h2>  
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>  
    </div>  
  
    <div class="input-container">
      <input type="text" class="input-text" placeholder="请输入待测域名..." v-model="inputValue" name="url" />  
      <button type="button" class="btn btn-outline-primary" @click="checkInput" name="detect-url">  
        <svg xmlns="http://www.w3.org/2000/svg" width="35px" height="35px" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">  
          <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"></path>  
        </svg>   
      </button>  
    </div>  

    <div class="result">  
      <div v-if="isFailed" class="result-failed">  
        <div class="result-content">  
          <svg-icon icon-class="failed" class="result-icon" />  
          <span class="result-status">检测失败</span>  
        </div>
        <p class="result-reason">{{ failureReason }}</p>  
      </div>  
      
      <div v-if="isSuccessed" class="result-success">  
        <div class="result-content">  
          <svg-icon   
            :icon-class="resultMessage === '危险' ? 'danger' : 'success'"   
            class="result-icon"   
          />  
          <span class="result-status">{{ resultMessage  }}</span>  
        </div>
        <div class="container"> 
          <table class="table table-sm" style="margin-bottom: 0;text-align: center;">  
            <thead>  
              <tr>  
                <th>检测子模型</th>  
                <th>检测结果</th>  
                <th>恶意概率</th>  
                <th>p值（可信度）</th>  
              </tr>  
            </thead>  
            <tbody>  
              <tr v-for="(result, model) in resultData" :key="model">  
                <td>{{ model }}</td>  
                <td v-if="result[0] === 0">安全</td>  
                <td v-else-if="result[0] === 1">危险</td>
                <td>{{ result[1] }}</td>  
                <td>{{ result[2].toFixed(4) }}</td> 
              </tr>  
            </tbody>  
          </table> 
        </div>  
      </div>  
    </div>
  </main>
</template> 

<script>  
import axios from 'axios';

export default {  
  data() {  
    return {  
      inputValue: '',  
      resultMessage: '',  
      isFailed: false,  
      isSuccessed: false,
      failureReason: '', 
      resultData: [],  
      apiBaseUrl: 'http://xxxx:5005' // 默认地址，防止配置读取失败
    };  
  },  
  created() {
    // 在组件创建时读取配置文件
    this.loadConfig();
  },
  methods: {  
    // 加载配置文件
    async loadConfig() {
      try {
        // 相对路径解析：从当前vue文件出发，向上找到new_flask目录下的config.ini
        // 路径解析：../../..//config.ini
        // 解释：从domain.vue所在目录(/vue/src/views/detect/)向上返回到项目根目录，再进入new_flask目录
        const response = await axios.get('/config.ini', {
          responseType: 'text'
        });
        
        // 解析INI格式内容
        const configContent = response.data;
        const lines = configContent.split('\n');
        let inApiSection = false;
        
        for (const line of lines) {
          const trimmedLine = line.trim();
          // 查找[api]部分
          if (trimmedLine === '[api]') {
            inApiSection = true;
            continue;
          }
          
          // 在[api]部分下查找baseUrl配置
          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=');
            if (parts.length >= 2) {
              this.apiBaseUrl = parts[1].trim();
              console.log('从配置文件加载API地址:', this.apiBaseUrl);
              break;
            }
          }
          
          // 遇到其他部分则退出查找
          if (inApiSection && trimmedLine.startsWith('[')) {
            break;
          }
        }
      } catch (error) {
        console.warn('加载配置文件失败，使用默认API地址:', error.message);
        // 继续使用默认地址
      }
    },
    
    checkInput() {  
      // 清除之前的结果  
      this.resultMessage = '';  
      this.isFailed = false;  
      this.isSuccessed = false;
      this.failureReason = '';  
  
      // 验证输入值  
      if (!this.inputValue) {  
        this.isFailed = true;  
        this.failureReason = '域名不可为空，请重新输入！';  
      } else if (!/^[A-Za-z0-9.-]*$/.test(this.inputValue)) {  
        this.isFailed = true;  
        this.failureReason = '域名格式不正确，域名中只能包含字母、数字、点、短横线！';  
      } else if (!this.inputValue.includes('.')) {  
        this.isFailed = true;  
        this.failureReason = '域名中必须包含至少一个点（.）！';  
      } else {  
        // 使用从配置文件读取的API地址  
        axios.post(`${this.apiBaseUrl}/api/detect_url`, { url: this.inputValue })  
          .then(response => {  
            const { status, result } = response.data;  
            if (status === '1') {  
              this.isSuccessed = true
              this.resultMessage = '危险';  
              this.resultData = result;  
            } else {  
              this.isSuccessed = true
              this.resultMessage = '安全';  
              this.resultData = result;  
            }  
          })  
          .catch(error => {  
            console.error(error);  
            this.isFailed = true;  
            this.failureReason = '请求后端时发生错误：' + error.message;  
          });  
      }  
    }  
  }  
};   
</script> 

<style scoped>  
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

.result-content {  
  display: flex;  
  justify-content: center;
  align-items: center;  
}  

.result-icon {  
  width: 40px;  
  height: 40px;  
  margin-right: 20px;  
  margin-top:5px;
}  

.result-status {  
  font-size: 30px;  
  color: #333;  
  font-weight: bold;  
}  
  
.result-reason {  
  font-size: 20px;  
  color: #666;  
  margin-top: 20px;  
  text-align: center;  
}  

.container {  
  display: flex;  
  justify-content: center;  
  padding: 5%;  
  box-sizing: border-box;  
} 

.table {  
  margin-top: 0px;   
  width: 100%;  
  max-width: 800px;  
  border-collapse: collapse;  
  margin-bottom: 0;  
  text-align: center;  
}

.table th,  
.table td {  
  border: 1px solid #ddd;  
  padding: 8px;  
}  
  
.table th {  
  background-color: #f2f2f2;  
}   
</style>
