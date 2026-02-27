<template>
  <div class="flow-debug">
    <el-card>
      <div slot="header">
        <h2>FlowViz 调试页面</h2>
      </div>

      <div class="debug-content">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-card>
              <div slot="header">
                <h3>连接测试</h3>
              </div>

              <el-button :loading="testingConnection" @click="testConnection">
                测试连接
              </el-button>

              <el-button :loading="testingStream" @click="testStream">
                测试流接口
              </el-button>

              <el-button @click="clearTokens">
                清除Token
              </el-button>

              <div v-if="connectionResult" class="result">
                <h4>连接结果:</h4>
                <pre>{{ JSON.stringify(connectionResult, null, 2) }}</pre>
              </div>
            </el-card>
          </el-col>

          <el-col :span="12">
            <el-card>
              <div slot="header">
                <h3>Token状态</h3>
              </div>

              <div class="token-info">
                <p><strong>localStorage Token:</strong> {{ tokenPreview }}</p>
                <p><strong>Token长度:</strong> {{ tokenLength }}</p>
                <p><strong>是否已登录:</strong> {{ isLoggedIn ? '是' : '否' }}</p>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <el-row :gutter="20" style="margin-top: 20px;">
          <el-col :span="24">
            <el-card>
              <div slot="header">
                <h3>简单分析测试</h3>
              </div>

              <el-form>
                <el-form-item label="测试文本">
                  <el-input
                    v-model="testText"
                    type="textarea"
                    :rows="4"
                    placeholder="输入测试文本..."
                  />
                </el-form-item>

                <el-form-item>
                  <el-button type="primary" :loading="runningTest" @click="runSimpleTest">
                    运行测试
                  </el-button>
                </el-form-item>
              </el-form>

              <div v-if="testResult" class="result">
                <h4>测试结果:</h4>
                <pre>{{ testResult }}</pre>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <el-row :gutter="20" style="margin-top: 20px;">
          <el-col :span="24">
            <el-card>
              <div slot="header">
                <h3>日志输出</h3>
              </div>

              <div class="log-output">
                <div v-for="(log, index) in logs" :key="index" class="log-line">
                  [{{ log.time }}] {{ log.message }}
                </div>
              </div>

              <el-button size="small" @click="clearLogs">
                清空日志
              </el-button>
            </el-card>
          </el-col>
        </el-row>
      </div>
    </el-card>
  </div>
</template>

<script>
import { flowvizApi } from '@/api/flowviz'

export default {
  name: 'FlowDebug',
  data() {
    return {
      testingConnection: false,
      testingStream: false,
      runningTest: false,
      connectionResult: null,
      testResult: null,
      logs: [],
      testText: `网络攻击者通过钓鱼邮件发送恶意附件，受害者打开附件后，恶意软件在系统中执行。攻击者使用Mimikatz工具窃取凭证，然后横向移动到其他主机。最终，他们从网络中窃取敏感数据。`
    }
  },

  computed: {
    tokenPreview() {
      const token = localStorage.getItem('token')
      if (!token) return '无'
      return token.substring(0, 20) + '...'
    },

    tokenLength() {
      const token = localStorage.getItem('token')
      return token ? token.length : 0
    },

    isLoggedIn() {
      return !!localStorage.getItem('token')
    }
  },

  mounted() {
    this.log('调试页面已加载')
  },

  methods: {
    log(message) {
      this.logs.unshift({
        time: new Date().toLocaleTimeString(),
        message
      })

      // 限制日志数量
      if (this.logs.length > 100) {
        this.logs = this.logs.slice(0, 100)
      }
    },

    clearLogs() {
      this.logs = []
    },

    clearTokens() {
      localStorage.removeItem('token')
      localStorage.removeItem('flowviz_token')
      this.log('已清除Token')
    },

    async testConnection() {
      this.testingConnection = true
      this.log('开始测试连接...')

      try {
        const result = await flowvizApi.testConnection()
        this.connectionResult = result
        this.log('连接测试成功')
      } catch (error) {
        this.connectionResult = { error: error.message }
        this.log('连接测试失败: ' + error.message)
      } finally {
        this.testingConnection = false
      }
    },

    async testStream() {
      this.testingStream = true
      this.log('开始测试流接口...')

      // 设置全局回调
      window.handleStreamData = (data) => {
        this.log(`收到事件: ${data.type}`)
        console.log('调试事件:', data)
      }

      try {
        await flowvizApi.testStream()
        this.log('流接口测试完成')
      } catch (error) {
        this.log('流接口测试失败: ' + error.message)
      } finally {
        this.testingStream = false
      }
    },

    async runSimpleTest() {
      this.runningTest = true
      this.log('开始简单分析测试...')

      // 设置全局回调
      window.handleStreamData = (data) => {
        this.log(`收到事件: ${data.type}`)

        if (data.type === 'error') {
          this.testResult = `错误: ${data.error}`
        }
      }

      window.onStreamComplete = () => {
        this.log('流式分析完成')
        this.testResult = '分析完成，请查看浏览器控制台'
      }

      try {
        await flowvizApi.analyzeStream({
          provider: 'openai',
          model: 'gpt-4',
          text: this.testText
        })

        this.log('分析请求已发送')
      } catch (error) {
        this.log('分析请求失败: ' + error.message)
        this.testResult = `请求失败: ${error.message}`
      } finally {
        this.runningTest = false
      }
    }
  }
}
</script>

<style scoped>
.flow-debug {
  padding: 20px;
}

.debug-content {
  margin-top: 20px;
}

.result {
  margin-top: 20px;
  background: #f5f5f5;
  padding: 10px;
  border-radius: 4px;
  max-height: 300px;
  overflow: auto;
}

.token-info {
  line-height: 1.8;
}

.log-output {
  background: #000;
  color: #0f0;
  padding: 10px;
  border-radius: 4px;
  font-family: monospace;
  height: 300px;
  overflow-y: auto;
  margin-bottom: 10px;
}

.log-line {
  border-bottom: 1px solid #333;
  padding: 2px 0;
}
</style>
