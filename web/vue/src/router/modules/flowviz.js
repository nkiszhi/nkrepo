// vue/src/api/flowviz.js
/**
 * FlowViz API模块
 */
export const flowvizApi = {
  /**
   * 获取Token并确保登录
   */
  async ensureLogin() {
    try {
      // 检查是否有token
      const token = localStorage.getItem('token')
      if (token) {
        console.log('✅ 已存在Token')
        return token
      }

      // 尝试自动登录
      console.log('🔑 尝试自动登录...')
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          username: 'admin',
          password: '123456'
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data && data.token) {
          localStorage.setItem('token', data.token)
          console.log('✅ 自动登录成功')
          return data.token
        }
      }

      console.warn('⚠️ 自动登录失败，使用模拟Token')
      // 创建模拟token（开发环境使用）
      const mockToken = 'flowviz-mock-token-' + Date.now()
      localStorage.setItem('token', mockToken)
      localStorage.setItem('flowviz_token', mockToken)
      console.log('⚠️ 使用模拟Token:', mockToken)
      return mockToken
    } catch (error) {
      console.error('登录失败:', error)

      // 创建模拟token（开发环境使用）
      if (process.env.NODE_ENV === 'development') {
        const mockToken = 'flowviz-mock-token-' + Date.now()
        localStorage.setItem('token', mockToken)
        localStorage.setItem('flowviz_token', mockToken)
        console.log('⚠️ 使用模拟Token:', mockToken)
        return mockToken
      }

      return null
    }
  },

  /**
   * 流式分析主接口 - 按照 FlowViz 原始项目格式
   */
  async analyzeStream(params) {
    const { input, provider = 'openai', model = 'gpt-4' } = params

    console.log('🚀 开始流式分析:', { provider, model, inputType: typeof input, inputLength: input.length })

    // 确保登录
    await this.ensureLogin()

    // 判断输入类型（URL 或文本）
    const isUrl = input.startsWith('http://') || input.startsWith('https://')

    // 构建请求体 - 完全按照 FlowViz 原始项目格式
    const requestBody = {
      provider: provider,
      model: model,
      system: '你是网络威胁情报分析方面的专家。请严格按照要求的JSON格式返回分析结果。'
    }

    // 添加 url 或 text 字段
    if (isUrl) {
      requestBody.url = input
      console.log('🌐 分析类型: URL')
    } else {
      requestBody.text = String(input).substring(0, 50000) // 限制长度
      console.log('📝 分析类型: 文本')
    }

    console.log('📦 请求体:', JSON.stringify(requestBody).substring(0, 200) + '...')

    const token = localStorage.getItem('token') || ''
    const url = '/flowviz/api/analyze-stream'

    return new Promise((resolve, reject) => {
      // 设置超时
      const timeout = 300000
      const controller = new AbortController()
      const timeoutId = setTimeout(() => {
        controller.abort()
        reject(new Error(`请求超时，超过${timeout / 1000}秒`))
      }, timeout)

      // 确保body是有效的JSON字符串
      let jsonBody
      try {
        jsonBody = JSON.stringify(requestBody)
      } catch (error) {
        console.error('❌ 请求体JSON序列化失败:', error)
        reject(new Error('请求体JSON序列化失败: ' + error.message))
        return
      }

      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'text/event-stream'
        },
        body: jsonBody,
        signal: controller.signal
      })
        .then(response => {
          clearTimeout(timeoutId)

          if (!response.ok) {
            return response.text().then(text => {
              throw new Error(`服务器错误 ${response.status}: ${text}`)
            })
          }

          // 处理流式响应
          const reader = response.body.getReader()
          const decoder = new TextDecoder('utf-8')

          function readStream() {
            reader.read().then(({ done, value }) => {
              if (done) {
                console.log('✅ 流式读取完成')
                resolve({ success: true })
                return
              }

              // 解码并处理数据块
              const chunk = decoder.decode(value, { stream: true })
              const lines = chunk.split('\n')

              for (const line of lines) {
                if (line.trim() === '') continue

                if (line.startsWith('data: ')) {
                  const dataStr = line.substring(6)

                  if (dataStr === '[DONE]') {
                    console.log('🏁 收到完成信号')
                    resolve({ success: true })
                    return
                  }

                  try {
                    const data = JSON.parse(dataStr)

                    // 通过全局回调函数发送到前端
                    if (window.handleStreamData) {
                      window.handleStreamData(data)
                    }
                  } catch (e) {
                    console.warn('⚠️ 解析事件失败:', e, '原始数据:', dataStr)
                  }
                }
              }

              // 继续读取
              readStream()
            })
              .catch(error => {
                console.error('❌ 流式读取错误:', error)
                reject(error)
              })
          }

          // 开始读取流
          readStream()
        })
        .catch(error => {
          clearTimeout(timeoutId)
          console.error('❌ 请求失败:', error)
          reject(error)
        })
    })
  },

  /**
   * 测试流接口
   */
  async testStream() {
    console.log('🔧 测试流接口')

    // 确保登录
    await this.ensureLogin()

    return new Promise((resolve, reject) => {
      const url = '/flowviz/api/test-stream'
      const controller = new AbortController()
      const timeout = 60000
      const timeoutId = setTimeout(() => {
        controller.abort()
        reject(new Error('测试流接口超时'))
      }, timeout)

      // 获取token
      const token = localStorage.getItem('token') || ''

      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'text/event-stream'
        },
        signal: controller.signal
      })
        .then(response => {
          clearTimeout(timeoutId)

          if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status}`)
          }

          const reader = response.body.getReader()
          const decoder = new TextDecoder('utf-8')

          function readTestStream() {
            reader.read().then(({ done, value }) => {
              if (done) {
                console.log('✅ 测试流完成')
                resolve({ success: true })
                return
              }

              const chunk = decoder.decode(value)
              const lines = chunk.split('\n')

              for (const line of lines) {
                if (line.trim() === '') continue

                if (line.startsWith('data: ')) {
                  const dataStr = line.substring(6)

                  if (dataStr === '[DONE]') {
                    reader.cancel()
                    resolve({ success: true })
                    return
                  }

                  try {
                    const data = JSON.parse(dataStr)

                    // 通过全局回调函数发送到前端
                    if (window.handleStreamData) {
                      window.handleStreamData(data)
                    }
                  } catch (e) {
                    console.error('解析测试事件失败:', e)
                  }
                }
              }

              readTestStream()
            })
          }

          readTestStream()
        })
        .catch(error => {
          clearTimeout(timeoutId)
          console.error('❌ 测试流请求失败:', error)
          reject(error)
        })
    })
  },

  /**
   * 测试连接
   */
  async testConnection() {
    try {
      // 确保登录
      await this.ensureLogin()

      const token = localStorage.getItem('token') || ''

      const response = await fetch('/flowviz/api/providers', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (!response.ok) {
        throw new Error(`HTTP错误: ${response.status}`)
      }

      const data = await response.json()
      console.log('✅ FlowViz 连接测试成功:', data)
      return { success: true, data }
    } catch (error) {
      console.error('❌ FlowViz 连接测试失败:', error)
      throw error
    }
  },

  /**
   * 测试OpenAI连接
   */
  async testOpenAIConnection() {
    console.log('🔧 测试OpenAI连接')

    try {
      // 确保登录
      await this.ensureLogin()

      const token = localStorage.getItem('token') || ''

      const response = await fetch('/flowviz/api/test-openai', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          provider: 'openai'
        })
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`HTTP错误 ${response.status}: ${errorText}`)
      }

      const data = await response.json()
      console.log('✅ OpenAI连接测试结果:', data)
      return data
    } catch (error) {
      console.error('❌ OpenAI连接测试失败:', error)
      throw error
    }
  },

  /**
   * 获取提供商列表
   */
  async getProviders() {
    // 确保登录
    await this.ensureLogin()

    const token = localStorage.getItem('token') || ''

    const response = await fetch('/flowviz/api/providers', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })

    if (!response.ok) {
      throw new Error(`获取提供商失败: ${response.status}`)
    }

    return await response.json()
  },

  /**
   * 调试测试后端
   */
  async debugTestBackend() {
    try {
      const response = await fetch('/flowviz/api/health')
      if (!response.ok) {
        throw new Error(`HTTP错误: ${response.status}`)
      }

      const data = await response.json()
      console.log('✅ 后端健康检查:', data)
      return { success: true, data }
    } catch (error) {
      console.error('❌ 后端调试测试失败:', error)
      throw error
    }
  },

  /**
   * 简单测试 - 发送纯文本测试
   */
  async simpleTest(text = '测试攻击流程分析') {
    console.log('🔧 简单测试:', text)

    return this.analyzeStream({
      input: text,
      provider: 'openai',
      model: 'gpt-4'
    })
  }
}
