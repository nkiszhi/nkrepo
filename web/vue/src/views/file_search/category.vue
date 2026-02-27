<template>
  <main>
    <div class="text-center">
      <h2 class="text-primary">样本类型查询</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
    </div>

    <div class="input-container">
      <input v-model="tableName" type="text" class="input-text" placeholder="请输入一个病毒类型进行查询" name="category">
      <button type="button" class="btn btn-outline-primary" name="detect-category" :disabled="isSearchElLoading" @click="searchVirus">
        <svg xmlns="http://www.w3.org/2000/svg" width="35px" height="35px" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">
          <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z" />
        </svg>
        <span v-if="isSearchElLoading" class="loading-text">查询中...</span>
      </button>
    </div>

    <div class="chart-wrapper" style="width:100%; height:200%;">
      <pie1-chart />
    </div>

    <div v-if="isSearchElLoading" class="search-status">
      <svg-icon icon-class="file_search" class="result-icon" />
      <span class="icon file-search" /> 样本搜索中，请稍等...
    </div>
    <div v-else-if="searchResults.length > 0" class="search-results">
      <h2>符合条件的样本如下：</h2>
      <ul class="flex-container">
        <li v-for="(sha256, index) in searchResults" :key="sha256" class="table-row" @mouseover="hoveringOver = sha256" @mouseleave="hoveringOver = null" @click="toggleDetails(sha256)">
          <div class="table-row-inner flex-row" style="width: 100%; margin: 0 auto;">
            <div class="table-cell sha256-cell">
              <span class="label">样本{{ index + 1 }}  {{ sha256 }}</span>
              <!-- 单条数据加载状态 -->
              <span v-if="loadingDetails[sha256]" class="detail-loading">（加载详情中...）</span>
            </div>
          </div>

          <div v-if="showDetails[sha256]" class="table-cell details-cell" style="width:90%; margin: 0 auto;">
            <p>MD5: {{ details[sha256]['MD5'] || 'N/A' }}</p>
            <p>SHA256: {{ details[sha256]['SHA256'] || 'N/A' }}</p>
            <p>类型: {{ details[sha256]['类型'] || 'N/A' }}</p>
            <p>平台: {{ details[sha256]['平台'] || 'N/A' }}</p>
            <p>家族: {{ details[sha256]['家族'] || 'N/A' }}</p>
            <p v-if="details[sha256] && details[sha256]['文件拓展名'] && details[sha256]['文件拓展名'] !== 'nan'">文件拓展名: {{ details[sha256]['文件拓展名'] }}</p>
            <p v-if="details[sha256] && details[sha256]['脱壳'] && details[sha256]['脱壳'] !== 'nan'">脱壳: {{ details[sha256]['脱壳'] }}</p>
            <p v-if="details[sha256] && details[sha256]['SSDEEP'] && details[sha256]['SSDEEP'] !== 'nan'">SSDEEP: {{ details[sha256]['SSDEEP'] }}</p>
            <button class="action-button" @click="(e) => downloadFile(e, sha256)">下载</button>
          </div>
        </li>
      </ul>
    </div>
    <div v-else-if="searchTriggered && searchResults.length === 0" class="search-empty">
      <p>未查询到符合条件的样本</p>
    </div>
  </main>
</template>

<script>
import axios from 'axios'
import Pie1Chart from '../dashboard/sample/admin/components/Pie1Chart.vue'

// 创建统一axios实例（10分钟超时 + 自动携带Token）
const apiService = axios.create({
  timeout: 600000, // 10分钟超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器：自动携带Token
apiService.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token') || sessionStorage.getItem('token')
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  error => Promise.reject(error)
)

export default {
  components: {
    Pie1Chart
  },
  data() {
    return {
      tableName: '',
      sha256s: [],
      details: {},
      hoveringOver: null, // 控制悬浮框的显示
      showDetails: {},
      searchQuery: '',
      isSearchElLoading: false, // 仅搜索时的加载状态
      loadingDetails: {}, // 单条数据的加载状态
      searchResults: [],
      apiBaseUrl: 'http://10.134.13.242:5005', // 兜底地址
      searchTriggered: false // 标记是否触发过搜索
    }
  },

  created() {
    // 组件创建时加载配置文件
    this.loadConfig()
  },
  methods: {
    // 加载配置文件获取API地址
    async loadConfig() {
      try {
        const response = await apiService.get('/config.ini', {
          responseType: 'text',
          timeout: 5000 // 配置读取单独设置5秒超时
        })

        // 解析INI格式内容
        const configContent = response.data
        const lines = configContent.split('\n')
        let inApiSection = false

        for (const line of lines) {
          const trimmedLine = line.trim()
          // 查找[api]部分
          if (trimmedLine === '[api]') {
            inApiSection = true
            continue
          }

          // 在[api]部分下查找baseUrl配置
          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=')
            if (parts.length >= 2) {
              this.apiBaseUrl = parts[1].trim()
              console.log('从配置文件加载API地址:', this.apiBaseUrl)
              break
            }
          }

          // 遇到其他部分则退出查找
          if (inApiSection && trimmedLine.startsWith('[')) {
            break
          }
        }

        // 配置文件未读取到baseUrl时，使用兜底地址
        if (!this.apiBaseUrl) {
          this.apiBaseUrl = 'http://10.134.13.242:5005'
          console.warn('配置文件未找到baseUrl，使用兜底地址')
        }
      } catch (error) {
        console.warn('加载配置文件失败，使用兜底地址:', error.message)
        this.apiBaseUrl = 'http://10.134.13.242:5005'
      }
    },

    searchVirus() {
      if (!this.tableName.trim()) {
        this.$message?.warning('请输入病毒类型后再查询') || alert('请输入病毒类型后再查询')
        return
      }
      this.isSearchElLoading = true
      this.searchResults = []
      this.searchTriggered = true

      // 使用从配置文件读取的API地址
      apiService.post(`${this.apiBaseUrl}/query_category`, { tableName: this.tableName.trim() })
        .then(response => {
          this.searchResults = response.data.sha256s || []
          if (this.$message) {
            this.$message.success(`查询成功，找到 ${this.searchResults.length} 个样本`)
          }
        })
        .catch(error => {
          console.error('查询失败:', error)
          let errMsg = '查询失败：'
          if (error.code === 'ECONNABORTED') {
            errMsg += '请求超时（已设置10分钟）'
          } else if (error.response?.status === 401) {
            errMsg += '登录状态失效，请重新登录'
            // 修复语法错误：拆分跳转逻辑
            localStorage.removeItem('token')
            sessionStorage.removeItem('token')
            if (this.$router) {
              this.$router.push('/login')
            } else {
              window.location.href = '/login'
            }
          } else if (error.response?.status === 500) {
            errMsg += '样本库未查询到'
          } else {
            errMsg += error.message || '未知错误'
          }
          this.$message?.error(errMsg) || alert(errMsg)
        })
        .finally(() => {
          this.isSearchElLoading = false
        })
    },

    toggleDetails(sha256) {
      // 切换详情显示状态
      this.$set(this.showDetails, sha256, !this.showDetails[sha256])
      // 如果是关闭详情，直接返回
      if (!this.showDetails[sha256]) return

      // 仅标记当前条目的加载状态，不影响全局
      this.$set(this.loadingDetails, sha256, true)

      // 使用从配置文件读取的API地址
      apiService.get(`${this.apiBaseUrl}/detail_category/${sha256}`)
        .then(response => {
          const result = response.data.query_result || {}
          this.$set(this.details, sha256, result)
          console.log(this.details[sha256])
        })
        .catch(error => {
          console.error('获取样本详情失败:', error)
          this.$message?.error('获取样本详情失败：' + error.message) || alert('获取样本详情失败：' + error.message)
          this.$set(this.showDetails, sha256, false)
        })
        .finally(() => {
          // 清除当前条目的加载状态
          this.$set(this.loadingDetails, sha256, false)
        })
    },

    // 修复后的下载方法（带Token + 语法错误修复 + 强制zip后缀）
    async downloadFile(e, sha256) {
      e.stopPropagation() // 阻止触发toggleDetails
      if (!sha256) return

      try {
        // 发起带Token的GET请求，获取文件流
        const response = await apiService.get(`${this.apiBaseUrl}/download_category/${sha256}`, {
          responseType: 'blob', // 关键：指定响应类型为二进制流
          timeout: 300000 // 下载超时5分钟
        })

        // 处理下载响应
        const blob = new Blob([response.data])
        const downloadUrl = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = downloadUrl

        // 强制设置zip后缀（解决后端返回文件名异常问题）
        let fileName = `${sha256}.zip`
        // 如果后端返回了正确的文件名，优先使用
        if (response.headers['content-disposition']) {
          const rawFileName = response.headers['content-disposition'].split('filename=')[1]
          if (rawFileName) {
            fileName = decodeURIComponent(rawFileName)
            // 确保文件名是zip后缀
            if (!fileName.endsWith('.zip')) {
              fileName = fileName + '.zip'
            }
          }
        }

        link.setAttribute('download', fileName)
        document.body.appendChild(link)
        link.click()

        // 清理资源
        document.body.removeChild(link)
        window.URL.revokeObjectURL(downloadUrl)
        this.$message?.success('下载已开始') || alert('下载已开始')
      } catch (error) {
        console.error('下载失败:', error)
        let errMsg = '下载失败'
        if (error.response?.status === 401) {
          errMsg = '登录状态失效，请重新登录后下载'
          // 修复语法错误：拆分跳转逻辑
          localStorage.removeItem('token')
          sessionStorage.removeItem('token')
          if (this.$router) {
            this.$router.push('/login')
          } else {
            window.location.href = '/login'
          }
        } else if (error.response?.status === 404) {
          errMsg = '文件不存在，无法下载'
        } else if (error.code === 'ECONNABORTED') {
          errMsg = '下载超时，请重试'
        } else {
          errMsg += '：' + error.message
        }
        this.$message?.error(errMsg) || alert(errMsg)
      }
    }
  }
}
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
  gap: 10px; /* 新增：按钮和输入框间距 */
}

.input-text {
  width: 800px;
  height: 50px;
  padding: 5px 15px; /* 优化内边距 */
  border-radius: 5px;
  border: 1px solid #ccc;
  box-sizing: border-box;
}

/* 按钮样式优化 */
.btn-outline-primary {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 5px 15px;
  border-radius: 5px;
  cursor: pointer;
  background-color: #fff;
  border: 1px solid #0d6efd;
  color: #0d6efd;
  transition: all 0.3s ease;
}

.btn-outline-primary:disabled {
  cursor: not-allowed;
  opacity: 0.7;
  background-color: #f8f9fa;
}

.loading-text {
  font-size: 14px;
  margin-left: 8px;
}

/* 单条数据加载状态样式 */
.detail-loading {
  font-size: 14px;
  color: #007BFF;
  margin-left: 10px;
}

.chart-wrapper {
    background: #fff;
    padding: 20px;
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
  margin-bottom: 10px; /* 列表项之间的间隔 */
  width:50%; /* 宽度根据需要调整，这里设为100%以填充容器宽度 */
  position: relative; /* 为可能的子元素定位做准备 */
  justify-content: center;
  transition: background-color 0.3s ease;
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

/* 空结果提示样式 */
.search-empty {
  text-align: center;
  font-size: 18px;
  color: #666;
  margin: 30px 0;
}
</style>
