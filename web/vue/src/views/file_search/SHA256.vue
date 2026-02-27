<template>
  <main>
    <div class="text-center">
      <h2 class="text-primary">样本SHA256查询</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
    </div>

    <div class="input-container">
      <input v-model="tableName" type="text" class="input-text" placeholder="请输入一个病毒SHA256进行查询" name="category">
      <button type="button" class="btn btn-outline-primary" name="detect-category" :disabled="isElLoading" @click="searchVirus">
        <svg xmlns="http://www.w3.org/2000/svg" width="35px" height="35px" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">
          <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z" />
        </svg>
        <span v-if="isElLoading" class="loading-text">查询中...</span>
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

      <div v-if="hasValidResults">
        <table class="file-info-table">
          <tbody>
            <tr>
              <th>查询结果</th>
              <th />
            </tr>
            <tr v-for="(value, key) in filteredQueryItems" :key="key">
              <td>{{ key.replace('_', ' ') }}：</td>
              <td>{{ value }}</td>
            </tr>
            <tr>
              <td colspan="2" class="download-btn-container">
                <button class="download-btn" @click="downloadFile(queryResult.SHA256)">下载文件</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>
</template>

<script>
import axios from 'axios'
import LineChart from '../dashboard/sample/admin/components/LineChart.vue'
import chartData from '@/data/chart_data.js'

// 创建统一axios实例
const apiService = axios.create({
  timeout: 600000,
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
    LineChart
  },
  data() {
    return {
      tableName: '',
      queryResult: {}, // 初始化为空对象，避免undefined报错
      SearchResult: null,
      lineChartData: chartData.lineChartData.total_amount,
      apiBaseUrl: 'http://10.134.13.242:5005',
      isElLoading: false
    }
  },
  computed: {
    filteredQueryItems() {
      // 直接使用this.queryResult（不再嵌套query_sha256）
      if (!this.queryResult || typeof this.queryResult !== 'object') {
        return {}
      }
      const validItems = {}
      Object.entries(this.queryResult).forEach(([key, value]) => {
        // 过滤无效值（包括空字符串）
        if (value !== 'nan' && value !== 'nan\r' && value !== null && value !== undefined && value !== '') {
          validItems[key] = value
        }
      })
      return validItems
    },
    hasValidResults() {
      // 直接判断filteredQueryItems是否有数据
      return Object.keys(this.filteredQueryItems).length > 0
    }
  },
  created() {
    this.loadConfig()
  },
  methods: {
    async loadConfig() {
      try {
        const response = await apiService.get('/config.ini', {
          responseType: 'text',
          timeout: 5000
        })
        const configContent = response.data
        const lines = configContent.split('\n')
        let inApiSection = false
        for (const line of lines) {
          const trimmedLine = line.trim()
          if (trimmedLine === '[api]') {
            inApiSection = true
            continue
          }
          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=')
            if (parts.length >= 2) {
              this.apiBaseUrl = parts[1].trim()
              console.log('从配置文件加载API地址:', this.apiBaseUrl)
              break
            }
          }
          if (inApiSection && trimmedLine.startsWith('[')) {
            break
          }
        }
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
      this.tableName = this.tableName.trim() // 去前后空格
      if (!this.tableName || this.tableName.length !== 64) {
        this.SearchResult = '请输入正确的SHA256'
        this.$message?.warning('请输入64位的SHA256值') || alert('请输入正确的SHA256')
        return
      }
      this.isElLoading = true
      this.SearchResult = null
      this.queryResult = {} // 清空旧数据

      apiService.post(`${this.apiBaseUrl}/query_sha256`, { tableName: this.tableName })
        .then(response => {
          let resultData = response.data
          if (typeof resultData === 'string') {
            try {
              resultData = JSON.parse(resultData)
            } catch (e) {
              console.error('Failed to parse JSON:', e)
              throw new Error('返回数据格式异常')
            }
          }

          // 直接取data字段（后端返回的详情数据）
          this.queryResult = resultData.data || {}
          console.log('查询结果:', this.queryResult)

          // 校验是否有有效数据
          if (Object.keys(this.queryResult).length === 0) {
            this.SearchResult = '查询成功，但未找到该样本的详细信息'
            this.$message?.warning('未找到该样本的详细信息') || alert('查询成功，但未找到该样本的详细信息')
          } else {
            this.SearchResult = '查询成功'
            this.$message?.success('查询成功') || alert('查询成功')
          }
        })
        .catch(error => {
          console.error('Error fetching data:', error)
          this.queryResult = {}
          let errMsg = '未查询到此样本'
          if (error.code === 'ECONNABORTED') {
            errMsg = '请求超时（已设置10分钟）'
          } else if (error.response?.status === 401) {
            errMsg = '登录状态失效，请重新登录'
            // 拆分跳转逻辑
            localStorage.removeItem('token')
            sessionStorage.removeItem('token')
            if (this.$router) {
              this.$router.push('/login')
            } else {
              window.location.href = '/login'
            }
          } else if (error.message.includes('Failed to parse JSON')) {
            errMsg = '后端返回数据格式异常'
          } else if (error.response?.data?.error) {
            errMsg = error.response.data.error // 显示后端返回的具体错误
          }
          this.SearchResult = errMsg
          this.$message?.error(errMsg) || alert(errMsg)
        })
        .finally(() => {
          this.isElLoading = false
        })
    },

    // 修复后的下载方法（带Token + 强制ZIP后缀）
    async downloadFile(sha256) {
      if (!sha256) {
        this.$message?.warning('SHA256值为空，无法下载') || alert('SHA256值为空，无法下载')
        return
      }

      this.isElLoading = true
      try {
        const response = await apiService.get(`${this.apiBaseUrl}/download_sha256/${sha256}`, {
          responseType: 'blob',
          timeout: 300000
        })

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

        document.body.removeChild(link)
        window.URL.revokeObjectURL(downloadUrl)
        this.$message?.success('下载已开始') || alert('下载已开始')
      } catch (error) {
        console.error('下载失败:', error)
        let errMsg = '下载失败'
        if (error.response?.status === 401) {
          errMsg = '登录状态失效，请重新登录后下载'
          // 拆分跳转逻辑
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
      } finally {
        this.isElLoading = false
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
  gap: 10px;
}

.input-text {
  width: 800px;
  height: 50px;
  padding: 5px 15px;
  border-radius: 5px;
  border: 1px solid #ccc;
  box-sizing: border-box;
}

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

.result-content {
  display: flex;
  justify-content: center;
  align-items: center;
}
.result-inner {
  display: flex;
  align-items: center;
  justify-content: center;
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
  text-justify:center;
}

.result-reason {
  font-size: 20px;
  color: #666;
  margin-top: 20px;
  text-align: center;
}

.file-info-table {
  width: 55%;
  margin: 30px auto;
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

.download-btn-container {
  text-align: right;
}

.download-btn {
  display: inline-block;
  width: 100%;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  padding: 10px 20px;
  font-size: 16px;
  font-weight: bold;
  transition: background-color 0.3s;
  cursor: pointer;
}

.download-btn:hover {
  background-color: #0056b3;
}
</style>
