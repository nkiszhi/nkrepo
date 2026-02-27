<template>
  <div class="function-detail-page" :class="{ 'fullscreen-mode': isFullscreen }">
    <!-- 顶部导航栏（全屏时隐藏） -->
    <div class="page-header" v-show="!isFullscreen">
      <el-button
        icon="el-icon-arrow-left"
        type="primary"
        @click="goBack"
        class="back-button"
        :disabled="isNavigating"
        size="small"
      >
        返回
      </el-button>
      <h1 class="page-title">{{ getPageTitle() }}</h1>
    </div>

    <!-- 加载状态（全屏时隐藏） -->
    <div v-if="loading && !isFullscreen" class="loading-container">
      <el-skeleton :rows="10" animated />
    </div>

    <!-- 错误状态（全屏时隐藏） -->
    <div v-else-if="error && !isFullscreen" class="error-container">
      <el-alert
        title="加载失败"
        :description="error"
        type="error"
        show-icon
      />
      <el-button type="primary" style="margin-top: 20px;" @click="loadFunctionDetail">
        重新加载
      </el-button>
    </div>

    <!-- 函数详情内容（全屏时隐藏） -->
    <div v-else-if="showFunctionContent" class="function-content">
      <!-- 功能描述卡片 -->
      <el-card class="description-card">
        <div slot="header" class="card-header">
          <span class="header-title">功能描述</span>
        </div>
        <div class="description-content">
          {{ functionData.summary }}
        </div>
      </el-card>

      <!-- ATT&CK技术卡片 -->
      <el-card v-if="hasTechniques" class="techniques-card">
        <div slot="header" class="card-header">
          <span class="header-title">ATT&CK 技术</span>
        </div>
        <div class="techniques-list">
          <el-tag
            v-for="tech in functionData.techniques"
            :key="tech.technique_id"
            type="danger"
            size="medium"
            style="margin-right: 10px; margin-bottom: 10px;"
          >
            {{ getTechniqueLabel(tech) }}
          </el-tag>
        </div>
      </el-card>

      <!-- 基本信息卡片 -->
      <el-card class="basic-info-card">
        <div slot="header" class="card-header">
          <span class="header-title">基本信息</span>
        </div>
        <el-descriptions :column="2" border>
          <el-descriptions-item label="根函数">
            <el-tag type="info">{{ getRootFunctionLabel() }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="文件名称">
            <span class="file-name">{{ functionData.file_name }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ formatDate(functionData.created_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="更新时间">
            {{ formatDate(functionData.updated_at) }}
          </el-descriptions-item>
          <el-descriptions-item label="子函数数量">
            <el-tag :type="getCountType(functionData.children_aliases_count)" size="small">
              {{ getCountLabel(functionData.children_aliases_count) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="技术数量">
            <el-tag :type="getCountType(functionData.technique_count)" size="small">
              {{ getCountLabel(functionData.technique_count) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="尝试次数">
            <el-tag type="warning" size="small">
              {{ getCountLabel(functionData.tries) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType()" size="small">
              {{ getStatusLabel() }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>
      </el-card>

      <!-- C++源代码卡片 -->
      <el-card class="code-card">
        <div slot="header" class="card-header">
          <span class="header-title">C++ 源代码</span>
          <div class="code-actions">
            <el-button
              v-if="hasCppCode"
              type="primary"
              icon="el-icon-document-copy"
              @click="copyCode"
              size="small"
              plain
              style="margin-right: 8px;"
            >
              复制代码
            </el-button>
            <el-button
              v-if="hasCppCode"
              type="primary"
              icon="el-icon-download"
              @click="downloadCode"
              size="small"
              plain
              style="margin-right: 8px;"
            >
              下载代码
            </el-button>
            <el-button
              v-if="hasCppCode"
              type="primary"
              :icon="isFullscreen ? 'el-icon-close' : 'el-icon-full-screen'"
              @click="toggleFullscreen"
              size="small"
              plain
            >
              {{ isFullscreen ? '退出全屏' : '全屏查看' }}
            </el-button>
          </div>
        </div>
        <div class="code-container" ref="codeContainer">
          <!-- 核心修改：v-html渲染安全高亮的代码 -->
          <pre v-if="highlightedCode" class="code-content cpp-code" v-html="highlightedCode"></pre>
          <div v-else class="empty-code">
            <el-empty description="暂无C++源代码" :image-size="100" />
          </div>
        </div>
      </el-card>

      <!-- 子函数别名卡片 -->
      <el-card v-if="hasChildrenAliases" class="children-aliases-card">
        <div slot="header" class="card-header">
          <span class="header-title">子函数别名</span>
          <el-badge :value="functionData.children_aliases_count" type="primary" />
        </div>
        <el-table :data="getChildrenAliasesData()" style="width: 100%">
          <el-table-column label="别名键" prop="0" width="200" />
          <el-table-column label="别名值" prop="1" show-overflow-tooltip />
        </el-table>
      </el-card>
    </div>
    
    <!-- 空状态（全屏时隐藏） -->
    <div v-else-if="!isFullscreen" class="empty-state">
      <el-empty description="未找到函数数据" />
      <el-button type="primary" @click="goBack" style="margin-top: 20px;">
        返回列表
      </el-button>
    </div>

    <!-- 全屏专用代码容器（居中显示） -->
    <div v-if="showFullscreenCode" class="fullscreen-code-wrapper">
      <div class="fullscreen-header">
        <span class="fullscreen-title">{{ getFullscreenTitle() }}</span>
        <el-button
          type="primary"
          icon="el-icon-close"
          @click="toggleFullscreen"
          size="small"
          circle
          class="fullscreen-close-btn"
        />
      </div>
      <div class="fullscreen-code-container">
        <pre class="fullscreen-code-content cpp-code" v-html="highlightedCode"></pre>
      </div>
    </div>
  </div>
</template>

<script>
import attckApi from '@/api/attck'
import hljs from 'highlight.js/lib/core'
import cpp from 'highlight.js/lib/languages/cpp'
import 'highlight.js/styles/github.css'
import DOMPurify from 'dompurify'

// 注册C++语言
hljs.registerLanguage('cpp', cpp)

export default {
  name: 'FunctionDetail',
  data() {
    return {
      loading: false,
      error: null,
      functionData: null,
      isNavigating: false,
      isFullscreen: false,
      highlightedCode: ''
    }
  },
  computed: {
    routeParams() {
      return this.$route.params
    },
    routeQuery() {
      return this.$route.query
    },
    processedCode() {
      if (!this.functionData || !this.functionData.cpp_code) {
        return ''
      }
      return this.functionData.cpp_code.replace(/，/g, ',')
    },
    // 简化模板条件判断：提取为computed
    showFunctionContent() {
      return this.functionData && !this.isFullscreen
    },
    hasTechniques() {
      return this.functionData && this.functionData.techniques && this.functionData.techniques.length > 0
    },
    hasCppCode() {
      return this.functionData && this.functionData.cpp_code
    },
    hasChildrenAliases() {
      if (!this.functionData || !this.functionData.children_aliases) {
        return false
      }
      return Object.keys(this.functionData.children_aliases).length > 0
    },
    showFullscreenCode() {
      return this.isFullscreen && this.functionData && this.functionData.cpp_code
    }
  },
  watch: {
    '$route': {
      handler(newRoute) {
        if (newRoute.query.id || (newRoute.query.file_name && newRoute.query.alias)) {
          this.loadFunctionDetail()
        }
      },
      immediate: false
    },
    functionData: {
      handler() {
        this.generateHighlightedCode()
      },
      immediate: true
    }
  },
  created() {
    this.loadFunctionDetail()
    // 监听系统全屏状态变化（防止手动按ESC退出状态不一致）
    document.addEventListener('fullscreenchange', this.handleFullscreenChange)
    document.addEventListener('webkitfullscreenchange', this.handleFullscreenChange)
    document.addEventListener('mozfullscreenchange', this.handleFullscreenChange)
    document.addEventListener('MSFullscreenChange', this.handleFullscreenChange)
  },
  beforeDestroy() {
    // 移除监听
    document.removeEventListener('fullscreenchange', this.handleFullscreenChange)
    document.removeEventListener('webkitfullscreenchange', this.handleFullscreenChange)
    document.removeEventListener('mozfullscreenchange', this.handleFullscreenChange)
    document.removeEventListener('MSFullscreenChange', this.handleFullscreenChange)
    // 退出全屏
    if (this.isFullscreen) {
      this.exitFullscreen()
    }
  },
  methods: {
    // 简化模板表达式：提取为方法
    getPageTitle() {
      if (this.functionData) {
        return this.functionData.alias
      } else {
        return '函数详情'
      }
    },
    getRootFunctionLabel() {
      if (this.functionData && this.functionData.root_function) {
        return this.functionData.root_function
      } else {
        return '无'
      }
    },
    getCountLabel(count) {
      if (count) {
        return count
      } else {
        return 0
      }
    },
    getStatusType() {
      if (this.functionData && this.functionData.status === 'ok') {
        return 'success'
      } else {
        return 'danger'
      }
    },
    getStatusLabel() {
      if (this.functionData && this.functionData.status) {
        return this.functionData.status
      } else {
        return 'unknown'
      }
    },
    getTechniqueLabel(tech) {
      return tech.technique_id + ': ' + tech.technique_name
    },
    getChildrenAliasesData() {
      if (this.functionData && this.functionData.children_aliases) {
        return Object.entries(this.functionData.children_aliases)
      } else {
        return []
      }
    },
    getFullscreenTitle() {
      if (this.functionData && this.functionData.alias) {
        return this.functionData.alias + ' - C++ 源代码'
      } else {
        return 'C++ 源代码'
      }
    },

    resetLocalStyle() {
      try {
        const pageEl = this.$el
        if (pageEl) {
          pageEl.style.overflow = 'auto'
          pageEl.style.position = 'relative'
        }
      } catch (error) {
        console.warn('重置页面样式时出错:', error)
      }
    },
    
    async loadFunctionDetail() {
      try {
        this.loading = true
        this.error = null
        
        const params = {}
        if (this.$route.query.id) {
          params.id = this.$route.query.id
        } else if (this.$route.query.file_name && this.$route.query.alias) {
          params.file_name = this.$route.query.file_name
          params.alias = this.$route.query.alias
        } else {
          this.error = '缺少必要的参数（需要id或file_name和alias）'
          this.loading = false
          return
        }
        
        const response = await attckApi.getFunctionDetail(params)
        if (response.success) {
          this.functionData = response.data
          this.generateHighlightedCode()
        } else {
          this.error = response.error || '获取函数详情失败'
        }
      } catch (error) {
        console.error('加载函数详情失败:', error)
        this.error = error.message || '加载函数详情失败'
      } finally {
        this.loading = false
        this.resetLocalStyle()
      }
    },
    
    goBack() {
      if (this.isNavigating) {
        return
      }
      
      this.isNavigating = true
      try {
        this.$router.go(-1)
      } catch (error) {
        console.error('返回失败:', error)
        this.$router.push('/')
      } finally {
        setTimeout(function() {
          this.isNavigating = false
        }.bind(this), 500)
      }
    },
    
    formatDate(dateStr) {
      if (!dateStr) {
        return '-'
      }
      try {
        const date = new Date(dateStr)
        if (isNaN(date.getTime())) {
          return dateStr
        }
        
        const pad = function(n) {
          return n.toString().padStart(2, '0')
        }
        return date.getFullYear() + '-' + pad(date.getMonth() + 1) + '-' + pad(date.getDate()) + ' ' + pad(date.getHours()) + ':' + pad(date.getMinutes())
      } catch (e) {
        return dateStr
      }
    },
    
    getCountType(count) {
      if (!count || count === 0) {
        return 'info'
      }
      if (count >= 10) {
        return 'success'
      }
      if (count >= 5) {
        return 'warning'
      }
      return 'primary'
    },
    
    generateHighlightedCode() {
      if (!this.functionData || !this.functionData.cpp_code) {
        this.highlightedCode = ''
        return
      }
      
      const fixedCode = this.functionData.cpp_code.replace(/，/g, ',')
      const highlightResult = hljs.highlight(fixedCode, { language: 'cpp' })
      const safeHtml = DOMPurify.sanitize(highlightResult.value)
      
      this.highlightedCode = safeHtml
    },
    
    copyCode() {
      if (!this.processedCode) {
        this.$message.warning('没有代码可以复制')
        return
      }
      
      navigator.clipboard.writeText(this.processedCode)
        .then(function() {
          this.$message.success('代码已复制到剪贴板')
        }.bind(this))
        .catch(function(err) {
          console.error('复制失败:', err)
          const textarea = document.createElement('textarea')
          textarea.value = this.processedCode
          textarea.style.position = 'fixed'
          textarea.style.opacity = '0'
          document.body.appendChild(textarea)
          textarea.select()
          try {
            document.execCommand('copy')
            this.$message.success('代码已复制到剪贴板')
          } catch (err) {
            this.$message.error('复制失败，请手动复制')
          } finally {
            document.body.removeChild(textarea)
          }
        }.bind(this))
    },
    
    downloadCode() {
      if (!this.processedCode) {
        this.$message.warning('没有代码可以下载')
        return
      }
      
      try {
        const blob = new Blob([this.processedCode], { type: 'text/plain;charset=utf-8' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        let downloadName = 'function'
        if (this.functionData && this.functionData.alias) {
          downloadName = this.functionData.alias
        }
        a.download = downloadName + '.cpp'
        a.style.display = 'none'
        document.body.appendChild(a)
        a.click()
        
        setTimeout(function() {
          document.body.removeChild(a)
          URL.revokeObjectURL(url)
          this.$message.success('代码下载成功')
        }.bind(this), 100)
      } catch (error) {
        this.$message.error('下载失败')
      }
    },
    
    // 优化：切换全屏（自定义全屏容器，而非原生全屏）
    toggleFullscreen() {
      this.isFullscreen = !this.isFullscreen
      // 全屏时滚动到顶部，禁止页面滚动
      if (this.isFullscreen) {
        document.body.style.overflow = 'hidden'
        window.scrollTo(0, 0)
      } else {
        document.body.style.overflow = 'auto'
        // 退出原生全屏（防止残留）
        this.exitFullscreen()
      }
    },
    
    // 退出原生全屏
    exitFullscreen() {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      } else if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen()
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen()
      } else if (document.msExitFullscreen) {
        document.msExitFullscreen()
      }
    },
    
    // 监听系统全屏状态变化
    handleFullscreenChange() {
      let isFullscreen = false
      if (document.fullscreenElement) {
        isFullscreen = true
      } else if (document.webkitFullscreenElement) {
        isFullscreen = true
      } else if (document.mozFullScreenElement) {
        isFullscreen = true
      } else if (document.msFullscreenElement) {
        isFullscreen = true
      }
      this.isFullscreen = isFullscreen
      if (!isFullscreen) {
        document.body.style.overflow = 'auto'
      }
    }
  }
}
</script>

<style scoped>
/* 原有样式完全保留 */
.function-detail-page {
  padding: 20px;
  background: #f0f2f5;
  min-height: calc(100vh - 60px);
  position: relative;
  transition: all 0.3s ease;
}

/* 全屏模式下隐藏非代码内容 */
.function-detail-page.fullscreen-mode {
  padding: 0;
  background: #fff;
  min-height: 100vh;
}

.page-header {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
  padding: 16px 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.back-button {
  margin-right: 16px;
}

.page-title {
  margin: 0;
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  flex: 1;
}

.loading-container,
.error-container,
.empty-state {
  padding: 40px;
  text-align: center;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  margin-top: 20px;
}

.function-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.description-card,
.techniques-card,
.basic-info-card,
.code-card,
.children-aliases-card {
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  background: white;
  overflow: hidden;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #ebeef5;
  background: #fafafa;
}

.header-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.code-actions {
  display: flex;
  align-items: center;
}

.description-content {
  padding: 20px;
  font-size: 14px;
  line-height: 1.6;
  color: #606266;
  white-space: pre-wrap;
}

.techniques-list {
  padding: 20px;
}

.basic-info-card .el-descriptions {
  margin: 0;
}

.file-name {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #666;
  word-break: break-all;
  background: #f5f5f5;
  padding: 2px 6px;
  border-radius: 4px;
}

/* 普通模式代码容器样式 */
.code-container {
  padding: 0;
  background: #f6f8fa;
  border-radius: 4px;
  overflow: hidden;
}

.code-content {
  margin: 0;
  padding: 20px;
  font-family: 'Courier New', Consolas, Monaco, 'Andale Mono', monospace;
  font-size: 13px;
  line-height: 1.6;
  color: #24292e;
  background: #f6f8fa;
  overflow-x: auto;
  white-space: pre;
  word-wrap: normal;
  max-height: 600px;
  overflow-y: auto;
  counter-reset: line;
}

.code-content .line {
  counter-increment: line;
  display: block;
  padding-left: 3.5em;
  position: relative;
}

.code-content .line::before {
  content: counter(line);
  position: absolute;
  left: 0;
  top: 0;
  width: 3em;
  padding-right: 0.5em;
  text-align: right;
  color: #999;
  border-right: 1px solid #e1e4e8;
  user-select: none;
}

.empty-code {
  padding: 40px 20px;
  text-align: center;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .function-detail-page {
    padding: 12px;
  }
  
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
    padding: 12px;
  }
  
  .back-button {
    margin-right: 0;
    margin-bottom: 8px;
  }
  
  .page-title {
    font-size: 24px;
  }
  
  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  
  .code-actions {
    align-self: flex-start;
  }
  
  .code-content {
    font-size: 12px;
    padding: 12px;
  }
}

/* 新增：全屏代码容器样式（核心优化） */
.fullscreen-code-wrapper {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: #f8f9fa;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  padding: 20px;
  box-sizing: border-box;
}

.fullscreen-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e1e4e8;
}

.fullscreen-title {
  font-size: 20px;
  font-weight: 600;
  color: #24292e;
}

.fullscreen-close-btn {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.fullscreen-code-container {
  flex: 1;
  background: #f6f8fa;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  max-width: 95%;
  width: 1200px; /* 限制最大宽度，更美观 */
  margin: 0 auto; /* 水平居中 */
}

.fullscreen-code-content {
  margin: 0;
  padding: 30px;
  font-family: 'Courier New', Consolas, Monaco, 'Andale Mono', monospace;
  font-size: 15px; /* 全屏时字体更大 */
  line-height: 1.8;
  color: #24292e;
  background: #f6f8fa;
  overflow-x: auto;
  white-space: pre;
  word-wrap: normal;
  height: 100%;
  box-sizing: border-box;
  counter-reset: line;
}

/* 全屏行号优化 */
.fullscreen-code-content .line {
  counter-increment: line;
  display: block;
  padding-left: 4em; /* 更大的行号间距 */
  position: relative;
}

.fullscreen-code-content .line::before {
  content: counter(line);
  position: absolute;
  left: 0;
  top: 0;
  width: 3.5em; /* 更宽的行号区域 */
  padding-right: 0.5em;
  text-align: right;
  color: #6a737d;
  border-right: 1px solid #e1e4e8;
  user-select: none;
  font-size: 14px;
}
</style>

<style scoped>
::v-deep .function-detail-page {
  overflow: auto;
  position: relative;
}

::v-deep .code-content,
::v-deep .fullscreen-code-content {
  user-select: text;
}

/* 高亮样式适配 */
::v-deep .hljs-keyword {
  color: #d73a49;
}
::v-deep .hljs-string {
  color: #032f62;
}
::v-deep .hljs-comment {
  color: #6a737d;
  font-style: italic;
}
::v-deep .hljs-number {
  color: #005cc5;
}
::v-deep .hljs-function {
  color: #6f42c1;
}
::v-deep .hljs-preprocessor {
  color: #005cc5;
}
::v-deep .hljs-operator {
  color: #6f42c1;
}
::v-deep .hljs-type {
  color: #22863a;
}
</style>