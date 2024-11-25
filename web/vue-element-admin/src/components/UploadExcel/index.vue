<template>
  <div>
    <div class="drop">
    <input type="file" @change="onFileChange" ref="fileInput" />
      <el-button :loading="loading" style="margin-left:16px;" size="mini" type="primary" @click="uploadFile">
        上传文件
      </el-button>
    </div>
    <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return {
      selectedFile: null,
      loading: false,
      errorMessage: null
    }
  },
  methods: {
    onFileChange(e) {
      this.selectedFile = e.target.files[0]
    },
    uploadFile() {
      if (!this.selectedFile) return

      this.loading = true
      this.errorMessage = null

      const formData = new FormData()
      formData.append('file', this.selectedFile)

      axios.post('http://10.134.2.27:5000/upload', formData)
        .then(response => {
          this.loading = false
          // 打印服务器响应
          console.log(response.data)
          // 根据需要更新你的应用状态或显示消息给用户
        })
        .catch(error => {
          this.loading = false
          this.errorMessage = '文件上传失败'
          console.error(error)
        })
    }
  }
}
</script>

<style scoped>
.excel-upload-input{
  display: none;
  z-index: -9999;
}
.drop{
  border: 2px dashed #bbb;
  width: 600px;
  height: 160px;
  line-height: 160px;
  margin: 0 auto;
  font-size: 24px;
  border-radius: 5px;
  text-align: center;
  color: #bbb;
  position: relative;
}
</style>
