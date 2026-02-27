/**
 * FlowViz API模块 - 严格模式版本
 */
export const flowvizApi = {
  /**
   * 确保登录状态
   */
  async ensureLogin() {
    try {
      const token = localStorage.getItem('token') || localStorage.getItem('flowviz_token')
      if (token) {
        console.log('✅ 已存在Token')
        return token
      }

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
          localStorage.setItem('flowviz_token', data.token)
          console.log('✅ 自动登录成功')
          return data.token
        }
      }

      console.warn('⚠️ 自动登录失败，使用模拟Token')
      const mockToken = 'flowviz-mock-token-' + Date.now()
      localStorage.setItem('token', mockToken)
      localStorage.setItem('flowviz_token', mockToken)
      console.log('⚠️ 使用模拟Token:', mockToken)
      return mockToken
    } catch (error) {
      console.error('登录失败:', error)

      const mockToken = 'flowviz-mock-token-' + Date.now()
      localStorage.setItem('token', mockToken)
      localStorage.setItem('flowviz_token', mockToken)
      console.log('⚠️ 使用模拟Token:', mockToken)
      return mockToken
    }
  },

  /**
   * 获取提供商列表
   */
  async getProviders() {
    await this.ensureLogin()

    const token = localStorage.getItem('token') || localStorage.getItem('flowviz_token') || ''

    try {
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
    } catch (error) {
      console.error('获取提供商失败:', error)
      return {
        success: false,
        providers: [
          { 
            id: 'openai', 
            name: 'OpenAI', 
            models: ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'], 
            configured: true,
            supports_strict_mode: true
          },
          { 
            id: 'claude', 
            name: 'Claude', 
            models: ['claude-3-5-sonnet-20241022'], 
            configured: true,
            supports_strict_mode: true
          }
        ]
      }
    }
  },

  /**
   * 流式分析 - 严格模式版本
   */
  async analyzeStream(params) {
    const { 
      input, 
      inputType = 'text', 
      provider = 'openai', 
      model = 'gpt-4o', 
      strictMode = true 
    } = params

    console.log('🚀 开始FlowViz严格模式分析:', {
      inputType,
      provider,
      model,
      strictMode,
      inputLength: input?.length || 0
    })

    // 确保登录
    await this.ensureLogin()

    // 构建请求体
    const requestBody = {
      provider: provider,
      model: model,
      strict_mode: strictMode
    }

    // 根据输入类型设置参数
    if (inputType === 'url') {
      requestBody.url = input
    } else {
      requestBody.text = String(input)
    }

    const token = localStorage.getItem('token') || localStorage.getItem('flowviz_token') || ''
    const url = '/flowviz/api/analyze-stream'

    console.log('📤 发送请求到:', url, '严格模式:', strictMode)

    return new Promise((resolve, reject) => {
      const timeout = 600000 // 10分钟超时
      let timeoutId = null
      let controller = null

      // 创建可取消的请求
      controller = new AbortController()

      // 设置超时
      timeoutId = setTimeout(() => {
        if (controller) {
          controller.abort()
        }
        console.warn('⏰ 请求超时（10分钟）')
        reject(new Error('分析超时，请稍后重试'))
      }, timeout)

      // 发送请求
      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive'
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal
      })
        .then(response => {
          clearTimeout(timeoutId)

          console.log('📥 收到响应:', response.status, response.statusText)

          if (!response.ok) {
            return response.text().then(text => {
              let errorMsg = `服务器错误 ${response.status}`
              if (response.status === 404) {
                errorMsg = 'API端点不存在'
              } else if (response.status === 401) {
                errorMsg = '未授权，请重新登录'
              } else if (response.status === 413) {
                errorMsg = '请求内容过长'
              }
              throw new Error(`${errorMsg}: ${text}`)
            })
          }

          if (!response.body) {
            throw new Error('响应体为空')
          }

          // 处理流式响应
          const reader = response.body.getReader()
          const decoder = new TextDecoder('utf-8')
          let buffer = ''

          const processStream = () => {
            reader.read().then(({ done, value }) => {
              if (done) {
                console.log('✅ 流式读取完成')
                resolve({ success: true })
                return
              }

              // 解码数据
              const chunk = decoder.decode(value, { stream: true })
              buffer += chunk

              // 按行分割处理SSE
              const lines = buffer.split('\n')
              buffer = lines.pop() || ''

              for (const line of lines) {
                if (line.trim() === '') continue

                if (line.startsWith('data: ')) {
                  const dataStr = line.substring(6)

                  // 处理完成信号
                  if (dataStr === '[DONE]') {
                    console.log('🏁 收到完成信号 [DONE]')

                    if (window.handleStreamData) {
                      window.handleStreamData({ type: 'complete' })
                    }
                    continue
                  }

                  try {
                    const data = JSON.parse(dataStr)
                    console.log('📨 解析事件:', data.type)

                    // 通过全局回调发送到前端
                    if (window.handleStreamData) {
                      window.handleStreamData(data)
                    }
                  } catch (e) {
                    console.warn('⚠️ 解析事件失败:', e, '原始数据:', dataStr.substring(0, 100))
                  }
                }
              }

              // 继续读取
              processStream()
            })
              .catch(error => {
                clearTimeout(timeoutId)
                console.error('❌ 流式读取错误:', error)

                if (window.handleStreamData) {
                  window.handleStreamData({
                    type: 'error',
                    error: error.message
                  })
                }

                reject(error)
              })
          }

          // 开始读取流
          processStream()
        })
        .catch(error => {
          clearTimeout(timeoutId)
          console.error('❌ 请求失败:', error)

          let errorMsg = error.message
          if (error.name === 'AbortError') {
            errorMsg = '请求超时'
          } else if (error.message.includes('Failed to fetch')) {
            errorMsg = '网络连接失败'
          } else if (error.message.includes('404')) {
            errorMsg = 'API端点不存在'
          }

          reject(new Error(errorMsg))
        })
    })
  },

  /**
   * 获取FlowViz配置
   */
  async getFlowVizConfig() {
    await this.ensureLogin()

    const token = localStorage.getItem('token') || localStorage.getItem('flowviz_token') || ''

    try {
      const response = await fetch('/flowviz/api/config', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (!response.ok) {
        throw new Error(`获取配置失败: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('获取FlowViz配置失败:', error)
      return {
        success: false,
        config: {
          strict_mode: true,
          default_provider: 'openai',
          default_model: 'gpt-4o'
        }
      }
    }
  },

  /**
   * 快速连接测试
   */
  async quickTest() {
    await this.ensureLogin()

    const token = localStorage.getItem('token') || localStorage.getItem('flowviz_token') || ''

    return new Promise((resolve, reject) => {
      const url = '/flowviz/health'
      const timeout = 10000
      const timeoutId = setTimeout(() => {
        reject(new Error('连接测试超时'))
      }, timeout)

      fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(response => {
          clearTimeout(timeoutId)

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`)
          }

          return response.json()
        })
        .then(data => {
          console.log('✅ 连接测试成功:', data)
          resolve({ success: true, data })
        })
        .catch(error => {
          clearTimeout(timeoutId)
          console.error('❌ 连接测试失败:', error)
          reject(new Error('连接测试失败: ' + error.message))
        })
    })
  },

  /**
   * 解析AI响应文本
   */
  parseAIResponse(text) {
    if (!text || text.trim() === '') {
      return { nodes: [], edges: [], error: '文本为空' }
    }

    try {
      let cleanedText = text.trim()

      // 尝试匹配JSON格式
      const jsonMatch = cleanedText.match(/```(?:json)?\s*([\s\S]*?)\s*```/)
      if (jsonMatch) {
        cleanedText = jsonMatch[1]
      }

      // 尝试匹配最外层的 {...}
      const braceMatch = cleanedText.match(/(\{[\s\S]*\})/)
      if (braceMatch) {
        cleanedText = braceMatch[1]
      }

      const parsedData = JSON.parse(cleanedText)

      // 验证基本结构
      if (!parsedData.nodes || !Array.isArray(parsedData.nodes)) {
        parsedData.nodes = []
      }

      if (!parsedData.edges || !Array.isArray(parsedData.edges)) {
        parsedData.edges = []
      }

      console.log(`✅ 解析成功: ${parsedData.nodes.length}节点, ${parsedData.edges.length}边`)
      return parsedData
    } catch (error) {
      console.error('❌ 解析AI响应失败:', error)

      return {
        nodes: [],
        edges: [],
        error: `解析失败: ${error.message}`
      }
    }
  }
}