<template>  
  <main> <!-- 新的根元素 -->  
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
      
    }
  },
  methods: {
   
    handleDrop(e) {
      e.stopPropagation()
      e.preventDefault()
      if (this.loading) return
      const files = e.dataTransfer.files
      if (files.length !== 1) {
        this.$message.error('只支持上传一个文件!')
        return
      }
      const rawFile = files[0] // only use files[0]
      this.upload(rawFile)
    },
    handleDragover(e) {
      e.stopPropagation()
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    },
    handleUpload() {
      this.$refs['file-upload-input'].click()
    },
    handleClick(e) {
      const files = e.target.files
      const rawFile = files[0] // only use files[0]
      if (!rawFile) return
      this.upload(rawFile)
    },
    async upload(rawFile) {    
      this.uploadResult = null;  
      this.loading = true;    
      const formData = new FormData();    
    formData.append('file', rawFile);    
    try {      
      const response = await fetch('http://10.134.2.27:5005/upload', {      
        method: 'POST',      
        body: formData,      
      });    
      console.log('Response status:', response.status);    
      console.log('Response headers:', response.headers); // 打印响应头
      this.$forceUpdate();
      

       // 检查响应状态码，如果非200-299之间，则视为失败  
      if (!response.ok) {      
        throw new Error('Failed to upload file: ' + response.statusText);      
      }  

       // 尝试解析JSON  
      const data = await response.json();  
      console.log('Response data:', data); // 打印响应数据  
      this.uploadResult = data;
      await this.fetchDetailAPI(); 
      this.$forceUpdate();  
    } catch (error) {      
      // 在catch块中打印出具体的错误信息  
      console.error('Error uploading file:', error);  
      if (error instanceof Error) {  
       // 如果是Error对象，可以打印出更详细的错误描述  
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
 
  axios.get(`http://10.134.2.27:5005/detection_API/${sha256}`, { params: { VT_API: VT_API } })  
    .then(detectionResponse => {  
      // 处理检测API的响应  
      if (Array.isArray(detectionResponse.data) && detectionResponse.data.length > 0) {  
        this.results = detectionResponse.data;  
      } else {  
        this.error = 'Unexpected response format from detection API';  
        this.isErrors = true;  
      }  
    })  
    .catch(error => {  
      // 捕获检测API的错误  
      this.isErrors = true;  
      console.error('Error fetching detection data:', error);  
      this.error = 'Error fetching data from detection API';  
    })  
    .finally(() => {  
      // 无论成功还是失败，都执行  
      this.checkAndUpdateUI();  
    });  
  
  axios.get(`http://10.134.2.27:5005/behaviour_API/${sha256}`, { params: { VT_API: VT_API } })  
    .then(behaviourResponse => {  
      // 处理行为API的响应  
      if (typeof behaviourResponse.data === 'object' && !Array.isArray(behaviourResponse.data)) {  
          this.behaviour_results = behaviourResponse.data;  
        } else if (typeof behaviourResponse.data === 'object' && behaviourResponse.data.message) {  
          // 处理后端返回的特定消息  
          this.isError = true;  
          this.behaviour_results = behaviourResponse.data; // 如果错误信息也需要显示在模板中，可以这样做  
        } else {  
          // 处理非预期的响应格式  
          this.isError = true;  
          this.error = 'Unexpected response format from behaviour API';  
        }  
      })  
      .catch(error => {  
        // 捕获请求错误  
        this.isError = true;  
        console.error('Error fetching behaviour data:', error);  
        this.error = 'Error fetching data from behaviour API';  
      }) 
    .finally(() => {  
      // 无论成功还是失败，都执行  
      this.checkAndUpdateUIs();  
    });  
  
  // 检查并更新UI的方法  
  this.checkAndUpdateUIs = () => {  
    if (!this.isLoadings) return; // 如果不是加载状态，则直接返回  
    this.$forceUpdate();  
    this.isLoadings = false;  
  }; 
  this.checkAndUpdateUI = () => {  
    if (!this.isLoading) return; // 如果不是加载状态，则直接返回  
    this.$forceUpdate();  
    this.isLoading = false;  
  };   
  
  // 初始设置为加载状态  
  this.isLoading = true;
  this.isLoadings = true;   
}
} 
 },  
};
</script>

<style scoped>
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
width: 60%; /* 表格宽度设置为100% */  
margin: 0 auto;
border: 1px solid #ccc; 
border-collapse: collapse; /* 合并边框 */ 
margin-top: 30px; 
}  

.file-info-table th,  
.file-info-table td {  
padding: 8px; /* 单元格内边距 */  
text-align: left; /* 文本左对齐 */  
border-bottom: 1px solid #ddd; /* 底部边框 */  
}  

.file-info-table tr:hover {  
background-color: #f5f5f5; /* 鼠标悬停时行背景色 */  
}  
 
/* 第二个表格样式 */  
.detection-result-table {  
width: 60%; /* 表格宽度设置为100% */  
margin: 0 auto;
border: 1px solid #ccc; 
border-collapse: collapse; /* 合并边框 */ 
margin-top: 30px; 
}  

.detection-result-table th,  
.detection-result-table td {  
padding: 8px; /* 单元格内边距 */  
text-align: left; /* 文本左对齐 */  
border-bottom: 1px solid #ddd; /* 底部边框 */  
}  

/* 结果列样式 */  
.detection-result-table td:last-child {  
text-align: center; /* 最后一列文本居中对齐 */  
}  

/* 成功的文本和图标样式 */  
.text-success {  
color: green; /* 文本颜色为绿色 */  
}  

.fas.fa-check {  
color: green; /* 图标颜色为绿色 */  
}  

/* 危险的文本和图标样式 */  
.text-danger {  
color: red; /* 文本颜色为红色 */  
}  

.fas.fa-exclamation-triangle {  
color: red; /* 图标颜色为红色 */  
}  
.centered-container {  
  display: flex;  
  flex-direction: column;  
  align-items: center; /* 水平居中 */  
  justify-content: center; /* 垂直居中 */  
  height: 90%; /* 占满整个视口高度 */  
  text-align: center; /* 确保文本内容也水平居中 */  
  padding: 20px; /* 添加一些内边距 */  
}  
  
/* 可能还需要为表格或其他元素添加一些样式来优化显示效果 */  
table {  
  width: 60%; /* 表格宽度设置为100% */  
  margin: 0 auto;
  border: 1px solid #ccc; 
  border-collapse: collapse; /* 合并边框 */ 
  margin-top: 30px;
  border-bottom: 1px solid 
}  
table th,  
table td {  
padding: 8px; /* 单元格内边距 */  
text-align: center; /* 文本左对齐 */  
border-bottom: 1px solid #ddd; /* 底部边框 */  
} 
.vt_table-row:hover {  
  background-color: #f0f0f0; /* 或者你喜欢的任何颜色 */  
  box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* 添加阴影效果 */  
  /* 如果需要，还可以添加其他样式，比如改变文字颜色等 */  
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
/* 错误消息的样式 */  
.error-message {  
  text-align: center;  
  color: red;  
} 
.isLoading {  
  text-align: center;    
}
.flex-container {  
  display: flex;  
  flex-direction: column; /* 假设你想垂直排列列表项 */  
  align-items: center; /* 垂直居中（如果需要水平居中，请调整flex-direction） */  
  list-style: none; /* 移除列表项前的默认标记 */  
  padding: 0; /* 移除默认的padding */  
  margin: 0; /* 移除默认的margin */
  text-align: left;  
} 
.list-row {  
  margin-bottom: 3px; /* 列表项之间的间隔 */  
  width:90%; /* 宽度根据需要调整，这里设为100%以填充容器宽度 */  
  position: relative; /* 为可能的子元素定位做准备 */   
} 
.list-row > * {  
  white-space: normal; /* 确保子元素也允许换行 */  
  word-break: break-all; /* 如果需要，也可以允许在任意字符间换行（谨慎使用） */  
  text-indent: 1em;
}
h3 {  
  color: black;  
  width: 90%;  
  margin: 0 auto;  
  text-align: left;  
  padding-top: 10px; /* 在元素顶部添加内边距 */  
  padding-bottom: 10px; /* 在元素底部添加内边距 */  
  /* 只设置上下边框 */  
  border-top: 1px solid grey;  
  border-bottom: 1px solid grey; 
} 
h4 {  
  color: black; /* 可选：如果你想要文本在蓝色背景上清晰可见，可以添加这个 */
  width:90%; 
  margin: 0 auto;
  text-align: left;
  padding-top: 10px; /* 在元素顶部添加内边距 */  
  padding-bottom: 0px; /* 在元素底部添加内边距 */
}
.behaviour_result_table {  
  width: 90%; 
  border-bottom: 0px solid;
  margin-top: 5px;
}
</style>
