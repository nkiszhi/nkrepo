<!-- vue/src/views/flowviz/components/FlowControlPanel.vue -->
<template>
  <div class="flow-control-panel">
    <el-form label-position="top" size="small">
      <!-- 输入模式切换 -->
      <el-form-item>
        <el-radio-group v-model="inputMode" @change="handleModeChange">
          <el-radio-button label="text">文本分析</el-radio-button>
          <el-radio-button label="url">URL分析</el-radio-button>
        </el-radio-group>
      </el-form-item>

      <!-- URL输入 -->
      <el-form-item v-if="inputMode === 'url'" label="文章URL">
        <el-input
          v-model="urlInput"
          placeholder="输入技术文章、威胁报告的URL..."
          clearable
          @keyup.enter.native="handleUrlSubmit"
        >
          <template slot="append">
            <el-button :loading="fetchingArticle" @click="fetchArticle">
              获取文章
            </el-button>
          </template>
        </el-input>
        <div class="url-tips">
          <span class="tip-item" @click="insertExampleUrl">
            <i class="el-icon-link" /> 插入示例URL
          </span>
        </div>

        <!-- 文章预览 -->
        <div v-if="articlePreview.title" class="article-preview">
          <div class="preview-header">
            <h4>{{ articlePreview.title }}</h4>
            <el-button
              size="mini"
              :disabled="!articlePreview.content"
              @click="useArticleForAnalysis"
            >
              使用此文章分析
            </el-button>
          </div>
          <div v-if="articlePreview.content" class="preview-content">
            <p>{{ truncateText(articlePreview.content, 200) }}</p>
          </div>
          <div v-if="articlePreview.metadata" class="preview-meta">
            <span>字数: {{ articlePreview.metadata.wordCount || 0 }}</span>
            <span>来源: {{ articlePreview.metadata.source || '未知' }}</span>
          </div>
        </div>
      </el-form-item>

      <!-- 文本输入 -->
      <el-form-item v-if="inputMode === 'text'" label="分析内容">
        <el-input
          v-model="textInput"
          type="textarea"
          :rows="8"
          placeholder="输入技术文章、威胁报告文本或JSON格式的技术数据..."
          resize="vertical"
          @keyup.ctrl.enter.native="handleTextSubmit"
        />
        <div class="input-tips">
          <span class="tip-item" @click="insertExample('text')">
            <i class="el-icon-document" /> 文本示例
          </span>
          <span class="tip-item" @click="insertExample('json')">
            <i class="el-icon-data-analysis" /> JSON示例
          </span>
          <span class="tip-item" @click="clearInput">
            <i class="el-icon-delete" /> 清空
          </span>
        </div>

        <!-- 文本统计 -->
        <div v-if="textInput" class="text-stats">
          <span>字数: {{ textInput.length }}</span>
          <span v-if="textInput.length > 5000" class="warning">
            ⚠️ 长文本分析可能需要较长时间
          </span>
        </div>
      </el-form-item>

      <!-- AI提供商选择 -->
      <el-form-item label="AI 提供商">
        <el-select
          v-model="localProvider"
          placeholder="选择 AI 提供商"
          style="width: 100%;"
          @change="handleProviderChange"
        >
          <el-option
            v-for="provider in availableProviders"
            :key="provider.id"
            :label="provider.displayName"
            :value="provider.id"
            :disabled="!provider.configured"
          />
        </el-select>
      </el-form-item>

      <!-- 模型选择 -->
      <el-form-item label="模型">
        <el-select
          v-model="localModel"
          placeholder="选择模型"
          style="width: 100%;"
          :disabled="!localProvider"
        >
          <el-option
            v-for="item in modelOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
      </el-form-item>

      <!-- 操作按钮 -->
      <el-form-item>
        <el-button
          type="primary"
          :loading="isLoading"
          :disabled="!canStartAnalysis || isStreaming"
          style="width: 100%; margin-bottom: 10px;"
          @click="handleStartAnalysis"
        >
          <i class="el-icon-video-play" style="margin-right: 5px;" />
          {{ isStreaming ? '分析中...' : '开始分析' }}
        </el-button>

        <el-button-group style="width: 100%;">
          <el-button
            :disabled="!hasData || isStreaming"
            style="width: 50%;"
            @click="handleClear"
          >
            <i class="el-icon-delete" style="margin-right: 5px;" />
            清空
          </el-button>

          <el-dropdown
            trigger="click"
            style="width: 50%;"
            @command="handleExport"
          >
            <el-button :disabled="!hasData || isStreaming" style="width: 100%;">
              <i class="el-icon-download" style="margin-right: 5px;" />
              导出
            </el-button>
            <el-dropdown-menu slot="dropdown">
              <el-dropdown-item command="png">PNG 图片</el-dropdown-item>
              <el-dropdown-item command="json">JSON 数据</el-dropdown-item>
              <el-dropdown-item command="attackflow">ATT&CK Flow v3</el-dropdown-item>
              <el-dropdown-item command="stix">STIX 格式</el-dropdown-item>
            </el-dropdown-menu>
          </el-dropdown>
        </el-button-group>
      </el-form-item>

      <!-- 高级设置 -->
      <el-collapse v-model="activeSettings">
        <el-collapse-item title="高级设置" name="advanced">
          <!-- 严格模式开关 -->
          <div class="setting-item">
            <el-checkbox v-model="strictMode" @change="handleStrictModeChange">
              严格提取模式
            </el-checkbox>
            <el-tooltip content="只提取文章中明确提到的技术指标，不进行推测" placement="top">
              <i class="el-icon-question" />
            </el-tooltip>
          </div>

          <!-- 流式渲染开关 -->
          <div class="setting-item">
            <el-checkbox v-model="streamingRender" @change="handleStreamingRenderChange">
              实时流式渲染
            </el-checkbox>
            <el-tooltip content="实时显示解析出的节点和边，提供更好的交互体验" placement="top">
              <i class="el-icon-question" />
            </el-tooltip>
          </div>

          <!-- 动画效果开关 -->
          <div class="setting-item">
            <el-checkbox v-model="enableAnimations" @change="handleAnimationChange">
              启用动画效果
            </el-checkbox>
            <el-tooltip content="节点和边添加时的动画效果" placement="top">
              <i class="el-icon-question" />
            </el-tooltip>
          </div>

          <!-- 最大节点数 -->
          <div class="setting-item">
            <span>最大节点数:</span>
            <el-input-number
              v-model="maxNodes"
              :min="10"
              :max="200"
              :step="10"
              size="mini"
              style="width: 100px; margin-left: 10px;"
            />
          </div>

          <!-- 布局方向 -->
          <div class="setting-item">
            <span>布局方向:</span>
            <el-select v-model="layoutDirection" size="mini" style="width: 100px; margin-left: 10px;">
              <el-option label="从上到下" value="TB" />
              <el-option label="从左到右" value="LR" />
              <el-option label="从右到左" value="RL" />
              <el-option label="从下到上" value="BT" />
            </el-select>
          </div>
        </el-collapse-item>
      </el-collapse>

      <!-- 进度显示 -->
      <div v-if="isStreaming" class="progress-container">
        <el-progress
          :percentage="progress.percentage"
          :status="getProgressStatus(progress.status)"
          :stroke-width="8"
          :show-text="false"
        />
        <div class="progress-message">
          <i class="el-icon-loading" style="margin-right: 5px;" />
          {{ progress.message || '正在分析...' }}
        </div>
        <div v-if="progress.stage" class="progress-stage">
          阶段: {{ progress.stage }}
        </div>
      </div>

      <!-- 快捷提示 -->
      <el-card shadow="never" class="tips-card">
        <div slot="header" class="clearfix">
          <span style="font-size: 14px; font-weight: bold;">💡 使用提示</span>
        </div>
        <div class="tips-content">
          <div v-if="inputMode === 'url'" class="tip-item">
            <i class="el-icon-link tip-icon" />
            <span class="tip-text">支持技术文章、威胁报告、博客等URL</span>
          </div>
          <div v-if="inputMode === 'text'" class="tip-item">
            <i class="el-icon-document tip-icon" />
            <span class="tip-text">支持纯文本描述攻击过程</span>
          </div>
          <div class="tip-item">
            <i class="el-icon-data-analysis tip-icon" />
            <span class="tip-text">支持JSON格式的技术数据（含IOC）</span>
          </div>
          <div class="tip-item">
            <i class="el-icon-s-data tip-icon" />
            <span class="tip-text">长文本自动分块处理，支持超长报告</span>
          </div>
          <div class="tip-item">
            <i class="el-icon-download tip-icon" />
            <span class="tip-text">分析完成后可导出多种格式</span>
          </div>
        </div>
      </el-card>
    </el-form>

    <!-- 连接测试按钮 -->
    <el-button
      type="info"
      size="small"
      style="width: 100%; margin-top: 15px;"
      :loading="testingConnection"
      @click="handleTestConnection"
    >
      <i class="el-icon-connection" style="margin-right: 5px;" />
      {{ testingConnection ? '测试中...' : '测试连接' }}
    </el-button>

    <!-- 连接测试结果 -->
    <div v-if="connectionResult" class="connection-result">
      <el-alert
        :title="connectionResult.title"
        :type="connectionResult.type"
        :closable="true"
        show-icon
        @close="clearConnectionResult"
      >
        {{ connectionResult.message }}
      </el-alert>
    </div>
  </div>
</template>

<script>
import { flowvizApi } from '@/api/flowviz'

export default {
  name: 'FlowControlPanel',
  props: {
    isLoading: {
      type: Boolean,
      default: false
    },
    isStreaming: {
      type: Boolean,
      default: false
    },
    provider: {
      type: String,
      default: 'openai'
    },
    model: {
      type: String,
      default: 'gpt-4o'
    },
    progress: {
      type: Object,
      default: () => ({
        stage: '',
        message: '',
        percentage: 0,
        status: ''
      })
    },
    hasData: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      inputMode: 'text', // 'text' 或 'url'
      urlInput: '',
      textInput: '',
      localProvider: this.provider,
      localModel: this.model,
      fetchingArticle: false,
      testingConnection: false,
      connectionResult: null,
      articlePreview: {
        title: '',
        content: '',
        metadata: null
      },
      activeSettings: [],
      strictMode: true,
      streamingRender: true,
      enableAnimations: true,
      maxNodes: 50,
      layoutDirection: 'TB',
      availableProviders: [
        {
          id: 'openai',
          displayName: 'OpenAI',
          models: ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
          defaultModel: 'gpt-4o',
          configured: true
        },
        {
          id: 'anthropic',
          displayName: 'Claude',
          models: ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
          defaultModel: 'claude-3-sonnet-20240229',
          configured: false
        }
      ],
      modelOptions: []
    }
  },

  computed: {
    canStartAnalysis() {
      if (this.inputMode === 'url') {
        return this.urlInput && this.isValidUrl(this.urlInput)
      } else {
        return this.textInput && this.textInput.trim().length > 0
      }
    }
  },

  watch: {
    provider: {
      immediate: true,
      handler(newVal) {
        this.localProvider = newVal
        this.updateModelOptions()
      }
    },
    model: {
      immediate: true,
      handler(newVal) {
        this.localModel = newVal
      }
    },
    localProvider(newVal) {
      this.updateModelOptions()
      // 发送变更事件给父组件
      this.$emit('provider-change', {
        provider: this.localProvider,
        model: this.localModel
      })
    },
    localModel(newVal) {
      this.$emit('provider-change', {
        provider: this.localProvider,
        model: this.localModel
      })
    }
  },

  mounted() {
    this.loadProviders()
    this.updateModelOptions()
  },

  methods: {
    /**
     * 获取进度条状态
     */
    getProgressStatus(status) {
      const statusMap = {
        'success': 'success',
        'error': 'exception',
        'warning': 'warning',
        'active': '',
        '': ''
      }
      return statusMap[status] || ''
    },

    /**
     * 加载提供商配置
     */
    async loadProviders() {
      try {
        const result = await flowvizApi.getProviders()
        if (result && result.providers) {
          this.availableProviders = result.providers
          this.updateModelOptions()

          // 如果当前选择的提供商不可用，切换到默认的
          const currentProvider = this.availableProviders.find(p => p.id === this.localProvider)
          if (!currentProvider || !currentProvider.configured) {
            this.localProvider = result.defaultProvider || 'openai'
          }
        }
      } catch (error) {
        console.error('加载提供商配置失败:', error)
        // 使用默认配置继续
      }
    },

    /**
     * 更新模型选项
     */
    updateModelOptions() {
      const provider = this.availableProviders.find(p => p.id === this.localProvider)
      if (provider && provider.models) {
        this.modelOptions = provider.models.map(model => ({
          label: this.formatModelName(model),
          value: model
        }))

        // 如果当前模型不在选项中，设置为默认模型
        if (!this.modelOptions.find(m => m.value === this.localModel)) {
          this.localModel = provider.defaultModel || provider.models[0]
        }
      } else {
        this.modelOptions = []
      }
    },

    /**
     * 格式化模型名称
     */
    formatModelName(model) {
      const nameMap = {
        'gpt-4o': 'GPT-4o (推荐)',
        'gpt-4-turbo': 'GPT-4 Turbo',
        'gpt-3.5-turbo': 'GPT-3.5 Turbo',
        'claude-3-opus-20240229': 'Claude 3 Opus',
        'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
        'claude-3-haiku-20240307': 'Claude 3 Haiku'
      }
      return nameMap[model] || model
    },

    /**
     * 检查URL是否有效
     */
    isValidUrl(url) {
      try {
        new URL(url)
        return url.startsWith('http://') || url.startsWith('https://')
      } catch {
        return false
      }
    },

    /**
     * 处理输入模式变更
     */
    handleModeChange(mode) {
      this.clearArticlePreview()
      this.$emit('mode-change', mode)
    },

    /**
     * 获取文章内容
     */
    async fetchArticle() {
      if (!this.isValidUrl(this.urlInput)) {
        this.$message.warning('请输入有效的URL')
        return
      }

      this.fetchingArticle = true

      try {
        const response = await flowvizApi.fetchArticle(this.urlInput)

        if (response.success && response.article) {
          this.articlePreview = {
            title: response.article.title || '无标题',
            content: response.article.content || '',
            metadata: {
              wordCount: response.article.wordCount,
              source: response.article.source,
              extractedAt: new Date().toLocaleString()
            }
          }

          this.$message.success(`成功获取文章: ${this.articlePreview.title}`)
        } else {
          this.$message.warning('获取文章失败，请检查URL或网络连接')
        }
      } catch (error) {
        console.error('获取文章失败:', error)
        this.$message.error('获取文章失败: ' + error.message)
      } finally {
        this.fetchingArticle = false
      }
    },

    /**
     * 使用文章进行分析
     */
    useArticleForAnalysis() {
      if (this.articlePreview.content) {
        this.textInput = this.articlePreview.content
        this.inputMode = 'text'
        this.$message.success('已加载文章内容到文本输入框')
      }
    },

    /**
     * 清空文章预览
     */
    clearArticlePreview() {
      this.articlePreview = {
        title: '',
        content: '',
        metadata: null
      }
    },

    /**
     * 处理开始分析
     */
    handleStartAnalysis() {
      if (this.inputMode === 'url') {
        if (!this.isValidUrl(this.urlInput)) {
          this.$message.warning('请输入有效的URL')
          return
        }
        this.$emit('start-analysis', {
          type: 'url',
          value: this.urlInput
        })
      } else {
        if (!this.textInput.trim()) {
          this.$message.warning('请输入分析内容')
          return
        }
        this.$emit('start-analysis', {
          type: 'text',
          value: this.textInput
        })
      }
    },

    /**
     * 处理URL提交
     */
    handleUrlSubmit() {
      if (this.canStartAnalysis) {
        this.handleStartAnalysis()
      }
    },

    /**
     * 处理文本提交
     */
    handleTextSubmit() {
      if (this.canStartAnalysis) {
        this.handleStartAnalysis()
      }
    },

    /**
     * 处理清空
     */
    handleClear() {
      this.textInput = ''
      this.urlInput = ''
      this.clearArticlePreview()
      this.$emit('clear-analysis')
    },

    /**
     * 清空输入
     */
    clearInput() {
      this.textInput = ''
      this.$message.success('已清空输入内容')
    },

    /**
     * 处理导出
     */
    handleExport(format) {
      this.$emit('export', format)
    },

    /**
     * 处理提供商变更
     */
    handleProviderChange() {
      this.updateModelOptions()
    },

    /**
     * 处理严格模式变更
     */
    handleStrictModeChange() {
      this.$emit('strict-mode-change', this.strictMode)
    },

    /**
     * 处理流式渲染变更
     */
    handleStreamingRenderChange() {
      this.$emit('streaming-render-change', this.streamingRender)
    },

    /**
     * 处理动画变更
     */
    handleAnimationChange() {
      this.$emit('animation-change', this.enableAnimations)
    },

    /**
     * 插入示例
     */
    insertExample(type) {
      let exampleText = ''

      switch (type) {
        case 'text':
          exampleText = `攻击者通过钓鱼邮件发送恶意Word文档，文档中包含恶意宏代码。受害者打开文档后，宏代码执行PowerShell脚本，下载并运行Cobalt Strike Beacon。攻击者使用Mimikatz窃取凭证，然后横向移动到域控制器，最终窃取敏感数据。`
          break
        case 'json':
          exampleText = JSON.stringify({
            data: {
              title: 'APT29攻击活动分析',
              indicators: {
                ip_addresses: ['192.168.1.100', '10.0.0.5'],
                domains: ['malicious.example.com', 'c2.example.org'],
                hashes: {
                  md5: 'd41d8cd98f00b204e9800998ecf8427e',
                  sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
                }
              },
              techniques: ['T1566.001', 'T1059.001', 'T1003.001', 'T1021.002']
            }
          }, null, 2)
          break
      }

      this.textInput = exampleText
      this.$message.success(`已插入${type}示例`)
    },

    /**
     * 插入示例URL
     */
    insertExampleUrl() {
      this.urlInput = 'https://blog.example.com/apt-threat-report'
      this.$message.success('已插入示例URL')
    },

    /**
     * 截断文本
     */
    truncateText(text, maxLength) {
      if (!text) return ''
      if (text.length <= maxLength) return text
      return text.substring(0, maxLength) + '...'
    },

    /**
     * 测试连接
     */
    async handleTestConnection() {
      this.testingConnection = true
      this.connectionResult = null

      try {
        const result = await flowvizApi.quickTest()

        this.connectionResult = {
          title: '连接成功',
          type: 'success',
          message: `后端服务正常运行，状态: ${result.data?.status || '正常'}`
        }

        this.$message.success('连接测试成功')
      } catch (error) {
        console.error('连接测试失败:', error)

        this.connectionResult = {
          title: '连接失败',
          type: 'error',
          message: error.message || '无法连接到后端服务'
        }

        this.$message.error('连接测试失败: ' + error.message)
      } finally {
        this.testingConnection = false
      }
    },

    /**
     * 清除连接结果
     */
    clearConnectionResult() {
      this.connectionResult = null
    }
  }
}
</script>

<style scoped>
.flow-control-panel {
  height: 100%;
  padding: 20px 15px;
  overflow-y: auto;
  background: #ffffff;
  box-sizing: border-box;
}

/* URL提示 */
.url-tips {
  display: flex;
  justify-content: flex-start;
  margin-top: 8px;
}

/* 输入提示 */
.input-tips {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
}

.tip-item {
  font-size: 12px;
  color: #409EFF;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: color 0.3s;
}

.tip-item:hover {
  color: #66b1ff;
  text-decoration: underline;
}

.tip-item i {
  margin-right: 4px;
}

/* 文章预览 */
.article-preview {
  margin-top: 15px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #ebeef5;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.preview-header h4 {
  margin: 0;
  font-size: 14px;
  color: #303133;
  flex: 1;
  margin-right: 10px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.preview-content {
  margin-bottom: 10px;
}

.preview-content p {
  margin: 0;
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
}

.preview-meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #909399;
}

/* 文本统计 */
.text-stats {
  margin-top: 8px;
  font-size: 12px;
  color: #909399;
  display: flex;
  justify-content: space-between;
}

.text-stats .warning {
  color: #E6A23C;
}

/* 设置项 */
.setting-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}

.setting-item:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

/* 进度容器 */
.progress-container {
  margin: 15px 0;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 6px;
  border-left: 3px solid #409EFF;
}

.progress-message {
  font-size: 12px;
  color: #606266;
  margin-top: 6px;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
}

.progress-stage {
  font-size: 11px;
  color: #909399;
  text-align: center;
  margin-top: 4px;
}

/* 提示卡片 */
.tips-card {
  margin-top: 20px;
}

.tips-content {
  font-size: 12px;
  color: #666;
  line-height: 1.6;
}

.tip-item {
  display: flex;
  align-items: flex-start;
  margin-bottom: 8px;
}

.tip-icon {
  color: #409EFF;
  margin-right: 8px;
  margin-top: 2px;
  font-size: 14px;
}

.tip-text {
  flex: 1;
}

/* 连接结果 */
.connection-result {
  margin-top: 15px;
}

/* 滚动条样式 */
.flow-control-panel::-webkit-scrollbar {
  width: 6px;
}

.flow-control-panel::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.flow-control-panel::-webkit-scrollbar-thumb {
  background: #c0c4cc;
  border-radius: 3px;
}

.flow-control-panel::-webkit-scrollbar-thumb:hover {
  background: #a0a4ac;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .flow-control-panel {
    padding: 15px 10px;
  }

  .input-tips,
  .url-tips {
    flex-direction: column;
    gap: 5px;
  }

  .tip-item {
    justify-content: center;
  }
}
</style>
