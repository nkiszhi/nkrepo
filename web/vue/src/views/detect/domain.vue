<template>
  <main>
    <div class="text-center">
      <h2 class="text-primary">基于可信度评估的多模型恶意域名检测</h2>
      <p class="text-muted text-secondary">南开大学反病毒实验室NKAMG</p>
    </div>

    <div class="input-container">
      <input v-model="inputValue" type="text" class="input-text" placeholder="请输入待测域名..." name="url">
      <button type="button" class="btn btn-outline-primary" name="detect-url" :disabled="isElLoading" @click="checkInput">
        <svg xmlns="http://www.w3.org/2000/svg" width="35px" height="35px" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">
          <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z" />
        </svg>
        <span v-if="isElLoading" class="loading-text">检测中...</span>
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
            :icon-class="resultElMessage === '危险' ? 'danger' : 'success'"
            class="result-icon"
          />
          <span class="result-status">{{ resultElMessage }}</span>
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
import axios from 'axios'

// 创建axios实例（统一配置10分钟超时+自动携带Token）
const apiService = axios.create({
  timeout: 600000, // 10分钟超时（600秒）
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
  data() {
    return {
      inputValue: '',
      resultElMessage: '',
      isFailed: false,
      isSuccessed: false,
      failureReason: '',
      resultData: [],
      apiBaseUrl: 'http://xxxx:5005', // 默认地址
      isElLoading: false // 新增：检测加载状态
    }
  },
  created() {
    // 加载配置文件
    this.loadConfig()
  },
  methods: {
    // 加载配置文件（优化容错+兜底地址）
    async loadConfig() {
      try {
        const response = await apiService.get('/config.ini', {
          responseType: 'text',
          timeout: 5000 // 配置读取单独设置5秒超时
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

        // 配置文件未读取到baseUrl时，使用兜底地址
        if (!this.apiBaseUrl || this.apiBaseUrl === 'http://xxxx:5005') {
          this.apiBaseUrl = 'http://10.134.13.242:5005'
          console.warn('配置文件未找到baseUrl，使用兜底地址:', this.apiBaseUrl)
        }
      } catch (error) {
        console.warn('加载配置文件失败，使用兜底地址:', error.message)
        this.apiBaseUrl = 'http://10.134.13.242:5005'
      }
    },

    checkInput() {
      // 清除之前的结果
      this.resultElMessage = ''
      this.isFailed = false
      this.isSuccessed = false
      this.failureReason = ''

      // 验证输入值
      if (!this.inputValue) {
        this.isFailed = true
        this.failureReason = '域名不可为空，请重新输入！'
        return
      }

      // 优化域名正则（支持更标准的域名格式）
      const domainRegex = /^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$/
      if (!domainRegex.test(this.inputValue)) {
        this.isFailed = true
        this.failureReason = '域名格式不正确，请输入合法的域名（如：example.com）！'
        return
      }

      // 检查API地址是否加载完成
      if (!this.apiBaseUrl) {
        this.isFailed = true
        this.failureReason = 'API地址未加载完成，请稍后重试！'
        return
      }

      // 设置加载状态
      this.isElLoading = true

      // 发送检测请求（使用配置的API地址+10分钟超时）
      apiService.post(`${this.apiBaseUrl}/api/detect_url`, { url: this.inputValue })
        .then(response => {
          const { status, result } = response.data
          // 验证返回数据格式
          if (typeof status !== 'string' || !result) {
            throw new Error('后端返回数据格式异常')
          }
          this.isSuccessed = true
          this.resultElMessage = status === '1' ? '危险' : '安全'
          this.resultData = result
        })
        .catch(error => {
          console.error('检测请求失败:', error)
          this.isFailed = true
          // 区分错误类型给出提示
          if (error.code === 'ECONNABORTED') {
            this.failureReason = '请求超时（已设置10分钟超时，请检查后端服务）'
          } else if (error.response?.status === 401) {
            this.failureReason = '登录状态失效，请重新登录后再检测'
          } else if (error.response?.status) {
            this.failureReason = `请求失败（状态码：${error.response.status}）：${error.message}`
          } else {
            this.failureReason = '请求后端时发生错误：' + (error.message || '未知错误')
          }
        })
        .finally(() => {
          // 重置加载状态
          this.isElLoading = false
        })
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
  font-size: 16px;
  box-sizing: border-box;
}

/* 新增：加载状态样式 */
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
  margin-top: 20px;
}

.result-icon {
  width: 40px;
  height: 40px;
  margin-right: 20px;
  margin-top: 5px;
}

.result-status {
  font-size: 30px;
  color: #333;
  font-weight: bold;
}

.result-reason {
  font-size: 20px;
  color: #dc3545; /* 失败提示改为红色，更醒目 */
  margin-top: 20px;
  text-align: center;
}

.container {
  display: flex;
  justify-content: center;
  padding: 2% 5%; /* 优化内边距 */
  box-sizing: border-box;
  margin-top: 20px;
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
  padding: 10px 8px; /* 优化单元格内边距 */
}

.table th {
  background-color: #f2f2f2;
  font-weight: 600;
}

/* 新增：表格行hover效果 */
.table tbody tr:hover {
  background-color: #f8f9fa;
}
</style>
