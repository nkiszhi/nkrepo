<template>
  <div class="app-container">
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span class="card-title">系统配置管理</span>
        <el-button
          type="primary"
          size="small"
          :loading="loading.save"
          style="float: right;"
          @click="saveAllConfigs"
        >
          保存所有配置
        </el-button>
      </div>

      <el-tabs v-model="activeTab" type="border-card">
        <!-- 账户设置 -->
        <el-tab-pane label="账户设置" name="account">
          <el-form ref="accountForm" :model="formData.fixed_user" label-width="120px" size="default">
            <el-form-item label="管理员账号" prop="username">
              <el-input
                v-model="formData.fixed_user.username"
                placeholder="请输入管理员账号"
                style="width: 300px;"
              />
            </el-form-item>

            <el-form-item label="新密码" prop="new_password">
              <el-input
                v-model="formData.fixed_user.new_password"
                type="password"
                placeholder="请输入新密码（留空则不修改）"
                style="width: 300px;"
                show-password
              />
            </el-form-item>

            <el-form-item label="确认密码" prop="confirm_password">
              <el-input
                v-model="formData.fixed_user.confirm_password"
                type="password"
                placeholder="请再次输入新密码"
                style="width: 300px;"
                show-password
              />
            </el-form-item>

            <el-form-item>
              <el-alert
                title="提示：修改密码后需要重新登录"
                type="info"
                :closable="false"
                style="width: 400px;"
              />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <!-- API配置 -->
        <el-tab-pane label="API配置" name="api">
          <el-form ref="apiForm" :model="formData.api" label-width="180px" size="default">
            <el-form-item label="VirusTotal API密钥" prop="vt_key">
              <el-input
                v-model="formData.api.vt_key"
                type="password"
                placeholder="请输入VirusTotal API密钥"
                style="width: 400px;"
                show-password
              />
              <el-button
                type="text"
                style="margin-left: 10px;"
                @click="showVtKey = !showVtKey"
              >
                {{ showVtKey ? '隐藏' : '显示' }}
              </el-button>
              <div v-if="showVtKey" class="api-key-show">
                {{ formData.api.vt_key || '未设置' }}
              </div>
            </el-form-item>

            <el-form-item>
              <el-alert
                title="VirusTotal API密钥用于文件检测功能"
                type="info"
                :closable="false"
                style="width: 400px;"
              />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <!-- AI提供商配置 -->
        <el-tab-pane label="AI提供商" name="ai">
          <el-tabs v-model="aiActiveTab" type="card">
            <!-- OpenAI配置 -->
            <el-tab-pane label="OpenAI" name="openai">
              <el-form ref="openaiForm" :model="formData.ai_provider" label-width="180px" size="default">
                <el-form-item label="API密钥" prop="openai_api_key">
                  <el-input
                    v-model="formData.ai_provider.openai_api_key"
                    type="password"
                    placeholder="请输入OpenAI API密钥"
                    style="width: 400px;"
                    show-password
                  />
                  <el-button
                    type="text"
                    style="margin-left: 10px;"
                    @click="showOpenaiKey = !showOpenaiKey"
                  >
                    {{ showOpenaiKey ? '隐藏' : '显示' }}
                  </el-button>
                  <div v-if="showOpenaiKey" class="api-key-show">
                    {{ formData.ai_provider.openai_api_key || '未设置' }}
                  </div>
                </el-form-item>

                <el-form-item label="API地址" prop="openai_base_url">
                  <el-input
                    v-model="formData.ai_provider.openai_base_url"
                    placeholder="请输入OpenAI API地址"
                    style="width: 400px;"
                  />
                </el-form-item>

                <el-form-item label="模型" prop="openai_model">
                  <el-select
                    v-model="formData.ai_provider.openai_model"
                    placeholder="请选择模型"
                    style="width: 300px;"
                  >
                    <el-option label="gpt-4o" value="gpt-4o" />
                    <el-option label="gpt-4-turbo" value="gpt-4-turbo" />
                    <el-option label="gpt-3.5-turbo" value="gpt-3.5-turbo" />
                  </el-select>
                </el-form-item>
              </el-form>
            </el-tab-pane>

            <!-- Claude配置 -->
            <el-tab-pane label="Claude" name="claude">
              <el-form ref="claudeForm" :model="formData.ai_provider" label-width="180px" size="default">
                <el-form-item label="API密钥" prop="claude_api_key">
                  <el-input
                    v-model="formData.ai_provider.claude_api_key"
                    type="password"
                    placeholder="请输入Claude API密钥"
                    style="width: 400px;"
                    show-password
                  />
                  <el-button
                    type="text"
                    style="margin-left: 10px;"
                    @click="showClaudeKey = !showClaudeKey"
                  >
                    {{ showClaudeKey ? '隐藏' : '显示' }}
                  </el-button>
                  <div v-if="showClaudeKey" class="api-key-show">
                    {{ formData.ai_provider.claude_api_key || '未设置' }}
                  </div>
                </el-form-item>

                <el-form-item label="API地址" prop="claude_base_url">
                  <el-input
                    v-model="formData.ai_provider.claude_base_url"
                    placeholder="请输入Claude API地址"
                    style="width: 400px;"
                  />
                </el-form-item>

                <el-form-item label="模型" prop="claude_model">
                  <el-select
                    v-model="formData.ai_provider.claude_model"
                    placeholder="请选择模型"
                    style="width: 300px;"
                  >
                    <el-option label="claude-3-opus-20240229" value="claude-3-opus-20240229" />
                    <el-option label="claude-3-5-sonnet-20240620" value="claude-3-5-sonnet-20240620" />
                    <el-option label="claude-3-haiku-20240307" value="claude-3-haiku-20240307" />
                  </el-select>
                </el-form-item>
              </el-form>
            </el-tab-pane>

            <!-- 通用设置 -->
            <el-tab-pane label="通用设置" name="general">
              <el-form ref="generalForm" :model="formData.ai_provider" label-width="180px" size="default">
                <el-form-item label="默认AI提供商" prop="default_ai_provider">
                  <el-radio-group v-model="formData.ai_provider.default_ai_provider">
                    <el-radio label="openai">OpenAI</el-radio>
                    <el-radio label="claude">Claude</el-radio>
                  </el-radio-group>
                </el-form-item>

                <el-form-item label="当前状态">
                  <el-tag :type="openaiStatus.type" style="margin-right: 10px;">
                    {{ openaiStatus.text }}
                  </el-tag>
                  <el-tag :type="claudeStatus.type">
                    {{ claudeStatus.text }}
                  </el-tag>
                </el-form-item>

                <el-form-item>
                  <el-alert
                    :title="configTips"
                    type="info"
                    :closable="false"
                    style="width: 500px;"
                  />
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>
        </el-tab-pane>

        <!-- 界面设置 -->
        <el-tab-pane label="界面设置" name="display">
          <el-form ref="displayForm" label-width="180px" size="default">
            <el-form-item label="主题颜色">
              <theme-picker style="float: left;height: 26px;margin: -3px 8px 0 0;" @change="themeChange" />
            </el-form-item>

            <el-form-item label="开启标签页视图">
              <el-switch v-model="tagsView" />
            </el-form-item>

            <el-form-item label="固定头部">
              <el-switch v-model="fixedHeader" />
            </el-form-item>

            <el-form-item label="侧边栏Logo">
              <el-switch v-model="sidebarLogo" />
            </el-form-item>

            <el-form-item>
              <el-alert
                title="提示：界面设置会立即生效"
                type="info"
                :closable="false"
                style="width: 400px;"
              />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <!-- 系统信息 -->
        <el-tab-pane label="系统信息" name="info">
          <el-descriptions class="margin-top" title="系统状态" :column="2" border>
            <el-descriptions-item label="配置来源">
              <el-tag :type="configSource === 'database' ? 'success' : 'warning'">
                {{ configSource === 'database' ? '数据库' : '配置文件' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="最后更新时间">
              {{ lastUpdate || '未知' }}
            </el-descriptions-item>
            <el-descriptions-item label="OpenAI配置状态">
              <el-tag :type="openaiStatus.type">
                {{ openaiStatus.text }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="Claude配置状态">
              <el-tag :type="claudeStatus.type">
                {{ claudeStatus.text }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="默认AI提供商">
              <el-tag>{{ formData.ai_provider.default_ai_provider || '未设置' }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="VT API状态">
              <el-tag :type="vtApiStatus.type">
                {{ vtApiStatus.text }}
              </el-tag>
            </el-descriptions-item>
          </el-descriptions>

          <div style="margin-top: 30px;">
            <el-button
              type="primary"
              :loading="loading.load"
              @click="loadConfigs"
            >
              刷新配置
            </el-button>
            <el-button
              type="warning"
              style="margin-left: 10px;"
              @click="resetToDefault"
            >
              重置为默认值
            </el-button>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script>
import { getConfigs, updateConfigs } from '@/api/settings'
import ThemePicker from '@/components/ThemePicker/index.vue'

export default {
  name: 'SystemSettings',
  components: { ThemePicker },
  data() {
    return {
      activeTab: 'account',
      aiActiveTab: 'openai',
      loading: {
        load: false,
        save: false
      },
      formData: {
        fixed_user: {
          username: '',
          new_password: '',
          confirm_password: ''
        },
        api: {
          vt_key: ''
        },
        ai_provider: {
          openai_api_key: '',
          openai_base_url: 'https://api.closeai-asia.com/v1',
          openai_model: 'gpt-4o',
          claude_api_key: '',
          claude_base_url: 'https://api.openai-proxy.org/anthropic',
          claude_model: 'claude-3-opus-20240229',
          default_ai_provider: 'openai'
        }
      },
      showVtKey: false,
      showOpenaiKey: false,
      showClaudeKey: false,
      configSource: 'database',
      lastUpdate: '',
      originalData: null
    }
  },
  computed: {
    tagsView: {
      get() {
        return this.$store.state.settings.tagsView
      },
      set(val) {
        this.$store.dispatch('settings/changeSetting', {
          key: 'tagsView',
          value: val
        })
      }
    },
    fixedHeader: {
      get() {
        return this.$store.state.settings.fixedHeader
      },
      set(val) {
        this.$store.dispatch('settings/changeSetting', {
          key: 'fixedHeader',
          value: val
        })
      }
    },
    sidebarLogo: {
      get() {
        return this.$store.state.settings.sidebarLogo
      },
      set(val) {
        this.$store.dispatch('settings/changeSetting', {
          key: 'sidebarLogo',
          value: val
        })
      }
    },
    openaiStatus() {
      const key = this.formData.ai_provider.openai_api_key
      return {
        type: key ? 'success' : 'warning',
        text: key ? '已配置' : '未配置'
      }
    },
    claudeStatus() {
      const key = this.formData.ai_provider.claude_api_key
      return {
        type: key ? 'success' : 'warning',
        text: key ? '已配置' : '未配置'
      }
    },
    vtApiStatus() {
      const key = this.formData.api.vt_key
      return {
        type: key ? 'success' : 'warning',
        text: key ? '已配置' : '未配置'
      }
    },
    configTips() {
      const tips = []
      if (!this.formData.ai_provider.openai_api_key && !this.formData.ai_provider.claude_api_key) {
        tips.push('请至少配置一个AI提供商的API密钥以使用攻击流分析功能')
      } else if (this.formData.ai_provider.default_ai_provider === 'openai' && !this.formData.ai_provider.openai_api_key) {
        tips.push('默认AI提供商为OpenAI，但未配置OpenAI API密钥')
      } else if (this.formData.ai_provider.default_ai_provider === 'claude' && !this.formData.ai_provider.claude_api_key) {
        tips.push('默认AI提供商为Claude，但未配置Claude API密钥')
      }
      return tips.join('；') || 'AI提供商配置正常'
    }
  },
  mounted() {
    this.loadConfigs()
  },
  methods: {
    themeChange(val) {
      this.$store.dispatch('settings/changeSetting', {
        key: 'theme',
        value: val
      })
    },
    async loadConfigs() {
      this.loading.load = true
      try {
        const response = await getConfigs()
        if (response.success && response.data) {
          // 保存原始数据用于比较
          this.originalData = JSON.parse(JSON.stringify(response.data))

          // 更新表单数据
          this.formData.fixed_user.username = response.data.fixed_user?.username || 'admin'
          this.formData.fixed_user.new_password = ''
          this.formData.fixed_user.confirm_password = ''

          this.formData.api.vt_key = response.data.api?.vt_key || ''

          const aiData = response.data.ai_provider || {}
          this.formData.ai_provider = {
            ...this.formData.ai_provider,
            ...aiData
          }

          this.configSource = response.data._source || 'database'
          this.lastUpdate = response.data._last_update || ''

          this.$message.success('配置加载成功')
        } else {
          this.$message.error('配置加载失败：' + (response.error || '未知错误'))
        }
      } catch (error) {
        console.error('加载配置失败:', error)
        this.$message.error('加载配置失败：' + error.message)
      } finally {
        this.loading.load = false
      }
    },

    async saveAllConfigs() {
      // 验证密码
      if (this.formData.fixed_user.new_password) {
        if (this.formData.fixed_user.new_password !== this.formData.fixed_user.confirm_password) {
          this.$message.error('两次输入的密码不一致')
          return
        }
        if (this.formData.fixed_user.new_password.length < 6) {
          this.$message.error('密码长度不能少于6位')
          return
        }
      }

      // 准备提交数据
      const submitData = {
        fixed_user: {
          username: this.formData.fixed_user.username
        },
        api: {
          vt_key: this.formData.api.vt_key
        },
        ai_provider: {
          openai_api_key: this.formData.ai_provider.openai_api_key,
          openai_base_url: this.formData.ai_provider.openai_base_url,
          openai_model: this.formData.ai_provider.openai_model,
          claude_api_key: this.formData.ai_provider.claude_api_key,
          claude_base_url: this.formData.ai_provider.claude_base_url,
          claude_model: this.formData.ai_provider.claude_model,
          default_ai_provider: this.formData.ai_provider.default_ai_provider
        }
      }

      // 如果有新密码，添加到提交数据
      if (this.formData.fixed_user.new_password) {
        submitData.fixed_user.new_password = this.formData.fixed_user.new_password
      }

      this.loading.save = true
      try {
        const response = await updateConfigs(submitData)
        if (response.success) {
          this.$message.success('配置保存成功')

          // 清除密码字段
          this.formData.fixed_user.new_password = ''
          this.formData.fixed_user.confirm_password = ''

          // 重新加载配置
          setTimeout(() => {
            this.loadConfigs()
          }, 1000)
        } else {
          this.$message.error('配置保存失败：' + (response.message || '未知错误'))
        }
      } catch (error) {
        console.error('保存配置失败:', error)
        this.$message.error('保存配置失败：' + error.message)
      } finally {
        this.loading.save = false
      }
    },

    resetToDefault() {
      this.$confirm('确定要重置为默认配置吗？', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        this.formData.fixed_user = {
          username: 'admin',
          new_password: '',
          confirm_password: ''
        }

        this.formData.api.vt_key = ''

        this.formData.ai_provider = {
          openai_api_key: '',
          openai_base_url: 'https://api.closeai-asia.com/v1',
          openai_model: 'gpt-4o',
          claude_api_key: '',
          claude_base_url: 'https://api.openai-proxy.org/anthropic',
          claude_model: 'claude-3-opus-20240229',
          default_ai_provider: 'openai'
        }

        this.$message.success('已重置为默认配置')
      })
    }
  }
}
</script>

<style scoped>
.app-container {
  padding: 20px;
}

.card-title {
  font-size: 18px;
  font-weight: bold;
}

.box-card {
  margin-bottom: 20px;
}

.api-key-show {
  margin-top: 5px;
  padding: 8px;
  background-color: #f5f7fa;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  font-family: monospace;
  word-break: break-all;
}

.el-tabs {
  margin-top: 20px;
}

.el-form-item {
  margin-bottom: 22px;
}

.el-descriptions {
  margin-top: 20px;
}
</style>
