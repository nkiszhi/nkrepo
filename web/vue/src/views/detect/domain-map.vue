<template>  
  <main> <!-- 新的根元素 -->  
    <div class="text-center">  
      <h2 class="text-primary">基于可信度评估的多模型恶意域名检测</h2>  
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>  
    </div>  
  
    <div class="input-container"> <!-- 新的 input 容器 -->  
      <input type="text" class="input-text" placeholder="请输入待测域名..." v-model="inputValue" name="url" />  
      <button type="button" class="btn btn-outline-primary" @click="checkInput" name="detect-url">  
        <!-- 嵌入搜索图标 -->  
        <svg xmlns="http://www.w3.org/2000/svg" width="35px" height="35px" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">  
          <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"></path>  
        </svg>   
      </button>  
    </div>  

    <div class="result" >  
      
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
      
  </main> <!-- 新的根元素结束 -->  
</template> 

<script>  
import axios from 'axios';

export default {  
  data() {  
    return {  
      inputValue: '', // 绑定输入框的值  
      resultMessage: '', // 临时保存判断结果，用于判断是否显示结果区域  
      isFailed: false, // 标记检测是否失败  
      isSuccessed: false,
      failureReason: '', // 保存失败的具体原因 
      resultData: [],  
    };  
  },  
  methods: {  
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
       // 如果输入有效，发送请求到 Flask 后端 
        
        axios.post('http://10.134.2.27:5005/api/detect', { url: this.inputValue })  
          .then(response => {  
            // 处理成功响应  
          const { status, result } = response.data;  
          if (status === '1') {  
            this.isSuccessed = true
            this.resultMessage = '危险'; // 只需要一个消息提示即可  
            this.resultData = result; // 保存后端返回的 message 字典数组  
          } else {  
            this.isSuccessed = true
            this.resultMessage = '安全'; // 只需要一个消息提示即可  
            this.resultData = result; // 保存后端返回的 message 字典数组  
            }  
          })  
          .catch(error => {  
            // 处理请求失败  
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
.result-reasons {  
  /* 原因文本的样式 */  
  font-size: 25px;  
  color: #13fa22;  
  margin-top: 10px; 
  text-align: center; 
} 
.table table-sm{
  text-justify:center;
}
.container {  
  /* 这是一个技巧，用于垂直居中 */  
  display: flex;  
  justify-content: center; /* 水平居中 */  
  height: 50vh; /* 使用视口高度来确保内容垂直居中 */   
  padding: 5%; /* 为内容添加一些内边距 */
  box-sizing: border-box; /* 确保 padding 不会影响到容器的总宽度和高度 */  
} 
.table {  
  margin-top: 0px;   
  width: 100%; /* 如果需要，可以设置一个具体的宽度 */  
  max-width: 800px; /* 限制最大宽度以适应不同屏幕 */  
  border-collapse: collapse; /* 合并表格边框 */  
  margin-bottom: 0; /* 继承自你的内联样式 */  
  text-align: center; /* 继承自你的内联样式 */  
}
.table th,  
.table td {  
  border: 1px solid #ddd; /* 添加边框 */  
  padding: 8px; /* 添加内边距 */  
}  
  
.table th {  
  background-color: #f2f2f2; /* 添加表头背景色 */  
}   
</style>