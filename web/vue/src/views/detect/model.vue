<template>  
  <main>  
    <div class="text-center">   
      <h2 class="text-primary">基于可信度评估的多模型恶意文件检测</h2>    
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>    
    </div>   
    <div>
      <input ref="file-upload-input" class="file-upload-input" type="file"  @change="handleClick">
      <div class="drop" @drop="handleDrop" @dragover="handleDragover">
        把待检文件拖到这里或
        <el-button :loading="loading" style="margin-left:0%;font-size: 20px;" size="mini" type="primary" @click="handleUpload">
          选择待检文件
        </el-button>
      </div>
      <div v-if="uploadResult">  
        <table class="file-info-table"> 
          <th>文件特征</th> 
          <th> </th> 
          <tr>  
            <td>文件名称：</td>  
            <td>{{ uploadResult.original_filename }}</td>  
          </tr>  
          <tr>  
            <td>文件大小：</td>  
            <td>{{ uploadResult.file_size }}</td>  
          </tr>  
          <tr v-for="(value, key) in uploadResult.query_result" :key="key" v-if="value !== 'nan' && value !== 'NaN'">  
            <td>{{ key.replace('_', ' ') }}：</td>  
            <td>{{ value }}</td>  
          </tr>  
        </table>   
      </div>    
    </div>
  </main>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      loading: false,
      uploadResult: null,
      apiBaseUrl: 'http://xxx2:5005', // 默认地址，防止配置读取失败
      isLoading: false,
      isLoadings: false,
      results: [],
      behaviour_results: {},
      isError: false,
      isErrors: false,
      error: ''
    }
  },
  created() {
    // 组件创建时加载配置文件
    this.loadConfig();
  },
  methods: {
    // 加载配置文件获取API地址
    async loadConfig() {
      try {
        // 与domain.vue保持一致的路径解析逻辑
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

    handleDrop(e) {
      e.stopPropagation();
      e.preventDefault();
      if (this.loading) return;
      const files = e.dataTransfer.files;
      if (files.length !== 1) {
        this.$message.error('只支持上传一个文件!');
        return;
      }
      const rawFile = files[0];
      this.upload(rawFile);
    },

    handleDragover(e) {
      e.stopPropagation();
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    },

    handleUpload() {
      this.$refs['file-upload-input'].click();
    },

    handleClick(e) {
      const files = e.target.files;
      const rawFile = files[0];
      if (!rawFile) return;
      this.upload(rawFile);
    },

    async upload(rawFile) {    
      this.uploadResult = null;  
      this.loading = true;    
      const formData = new FormData();    
      formData.append('file', rawFile);    
      
      try {      
        // 使用从配置文件读取的API地址
        const response = await fetch(`${this.apiBaseUrl}/upload`, {      
          method: 'POST',      
          body: formData,      
        });    
        
        console.log('Response status:', response.status);    
        console.log('Response headers:', response.headers);
        this.$forceUpdate();
      
        if (!response.ok) {      
          throw new Error('Failed to upload file: ' + response.statusText);      
        }  

        const data = await response.json();  
        console.log('Response data:', data);  
        this.uploadResult = data;
        await this.fetchDetailAPI(); 
        this.$forceUpdate();  
      } catch (error) {      
        console.error('Error uploading file:', error);  
        if (error instanceof Error) {  
          console.error('Error message:', error.message);  
          console.error('Error stack:', error.stack);  
        }  
        this.$message.error('文件上传失败！');    
      } finally {    
        this.loading = false;    
      }   
    },

    fetchDetailAPI() {  
      this.isLoading = true;
      this.results = []; 
      this.behaviour_results = {};
      this.isError = false;
      this.isErrors = false;
      
      if (this.uploadResult && this.uploadResult.query_result && this.uploadResult.VT_API) {  
        const sha256 = this.uploadResult.query_result.SHA256;  
        const VT_API = this.uploadResult.VT_API;  
        console.log('sha256:', sha256);  
        console.log('VT_API:', VT_API);  

        // 使用从配置文件读取的API地址
        axios.get(`${this.apiBaseUrl}/detection_API/${sha256}`, { params: { VT_API: VT_API } })  
          .then(detectionResponse => {  
            if (Array.isArray(detectionResponse.data) && detectionResponse.data.length > 0) {  
              this.results = detectionResponse.data;  
            } else {  
              this.error = 'Unexpected response format from detection API';  
              this.isErrors = true;  
            }  
          })  
          .catch(error => {  
            this.isErrors = true;  
            console.error('Error fetching detection data:', error);  
            this.error = 'Error fetching data from detection API';  
          })  
          .finally(() => {  
            this.checkAndUpdateUI();  
          });  

        // 使用从配置文件读取的API地址
        axios.get(`${this.apiBaseUrl}/behaviour_API/${sha256}`, { params: { VT_API: VT_API } })  
          .then(behaviourResponse => {  
            if (typeof behaviourResponse.data === 'object' && !Array.isArray(behaviourResponse.data)) {  
              this.behaviour_results = behaviourResponse.data;  
            } else if (typeof behaviourResponse.data === 'object' && behaviourResponse.data.message) {  
              this.isError = true;  
              this.behaviour_results = behaviourResponse.data;  
            } else {  
              this.isError = true;  
              this.error = 'Unexpected response format from behaviour API';  
            }  
          })  
          .catch(error => {  
            this.isError = true;  
            console.error('Error fetching behaviour data:', error);  
            this.error = 'Error fetching data from behaviour API';  
          }) 
          .finally(() => {  
            this.checkAndUpdateUIs();  
          });  

        this.checkAndUpdateUIs = () => {  
          if (!this.isLoadings) return;  
          this.$forceUpdate();  
          this.isLoadings = false;  
        }; 

        this.checkAndUpdateUI = () => {  
          if (!this.isLoading) return;  
          this.$forceUpdate();  
          this.isLoading = false;  
        };   

        this.isLoading = true;
        this.isLoadings = true;   
      }
    } 
  },  
};
</script>

<style scoped>
/* 样式部分保持不变 */
.file-upload-input{
  display: none;
  z-index: -9999;
}
.drop{
  border: 2px dashed #bbb;
  width: 60%;
  height: 10%;
  line-height: 160px;
  margin: 0 auto;
  font-size: 24px;
  border-radius: 5px;
  text-align: center;
  color: #bbb;
  position: relative;
}

.file-info-table {  
width: 60%;  
margin: 0 auto;
border: 1px solid #ccc; 
border-collapse: collapse; 
margin-top: 30px; 
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

.detection-result-table {  
width: 60%;  
margin: 0 auto;
border: 1px solid #ccc; 
border-collapse: collapse; 
margin-top: 30px; 
}  

.detection-result-table th,  
.detection-result-table td {  
padding: 8px;  
text-align: left;  
border-bottom: 1px solid #ddd;  
}  

.detection-result-table td:last-child {  
text-align: center;  
}  

.text-success {  
color: green;  
}  

.fas.fa-check {  
color: green;  
}  

.text-danger {  
color: red;  
}  

.fas.fa-exclamation-triangle {  
color: red;  
}  

.centered-container {  
  display: flex;  
  flex-direction: column;  
  align-items: center;  
  justify-content: center;  
  height: 90%;  
  text-align: center;  
  padding: 20px;  
}  

table {  
  width: 60%;  
  margin: 0 auto;
  border: 1px solid #ccc; 
  border-collapse: collapse; 
  margin-top: 30px;
  border-bottom: 1px solid 
}  

table th,  
table td {  
padding: 8px;  
text-align: center;  
border-bottom: 1px solid #ddd;  
} 

.vt_table-row:hover {  
  background-color: #f0f0f0;  
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);  
}

.red-text {  
  color: red;  
}  

.black-text {  
  color: black;  
}  

.gray-text {  
  color: gray;  
}

.error-message {  
  text-align: center;  
  color: red;  
} 

.isLoading {  
  text-align: center;    
}

.flex-container {  
  display: flex;  
  flex-direction: column;  
  align-items: center;  
  list-style: none;  
  padding: 0;  
  margin: 0;  
  text-align: left;  
} 

.list-row {  
  margin-bottom: 3px;  
  width:90%;  
  position: relative;   
} 

.list-row > * {  
  white-space: normal;  
  word-break: break-all;  
  text-indent: 1em;
}

h3 {  
  color: black;  
  width: 90%;  
  margin: 0 auto;  
  text-align: left;  
  padding-top: 10px;  
  padding-bottom: 10px;  
  border-top: 1px solid grey;  
  border-bottom: 1px solid grey; 
} 

h4 {  
  color: black; 
  width:90%; 
  margin: 0 auto;
  text-align: left;
  padding-top: 10px;  
  padding-bottom: 0px;  
}

.behaviour_result_table {  
  width: 90%; 
  border-bottom: 0px solid;
  margin-top: 5px;
}
</style>