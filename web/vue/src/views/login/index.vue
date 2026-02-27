<template>
  <div class="login-container">
    <el-form ref="loginForm" :model="loginForm" :rules="loginRules" class="login-form" autocomplete="on" label-position="left">
      <div class="title-container">
        <h4 class="title">基于可信度评估的多模型恶意代码检测系统</h4>
        <h4 class="title">欢迎您</h4>
      </div>

      <el-form-item prop="username">
        <span class="svg-container">
          <svg-icon icon-class="user" />
        </span>
        <el-input
          ref="username"
          v-model="loginForm.username"
          placeholder="Username"
          name="username"
          type="text"
          tabindex="1"
          autocomplete="on"
        />
      </el-form-item>

      <el-tooltip v-model="capsTooltip" content="Caps lock is On" placement="right" manual>
        <el-form-item prop="password">
          <span class="svg-container">
            <svg-icon icon-class="password" />
          </span>
          <el-input
            :key="passwordType"
            ref="password"
            v-model="loginForm.password"
            :type="passwordType"
            placeholder="Password"
            name="password"
            tabindex="2"
            autocomplete="on"
            @keyup.native="checkCapslock"
            @blur="capsTooltip = false"
            @keyup.enter.native="handleLogin"
          />
          <span class="show-pwd" @click="showPwd">
            <svg-icon :icon-class="passwordType === 'password' ? 'eye' : 'eye-open'" />
          </span>
        </el-form-item>
      </el-tooltip>

      <el-button :loading="loading || configElLoading" type="primary" style="width:100%;margin-bottom:30px;" :disabled="configElLoading" @click.native.prevent="handleLogin">
        <span v-if="configElLoading">加载配置中...</span>
        <span v-else>登录</span>
      </el-button>
    </el-form>
  </div>
</template>

<script>
import axios from 'axios'
import router from '@/router'
import store from '@/store'
import { asyncRoutes } from '@/router' // 导入异步路由

// 创建基础axios实例（仅用于加载配置）
const configApi = axios.create({
  timeout: 5000 // 配置加载超时5秒
})

export default {
  name: 'Login',
  data() {
    const validateUsername = (rule, value, callback) => {
      if (!value.trim()) {
        callback(new Error('请输入用户名'))
      } else {
        callback()
      }
    }
    const validatePassword = (rule, value, callback) => {
      if (!value.trim()) {
        callback(new Error('请输入密码'))
      } else if (value.length < 6) {
        callback(new Error('密码长度不能小于6位'))
      } else {
        callback()
      }
    }
    return {
      loginForm: {
        username: 'admin',
        password: ''
      },
      loginRules: {
        username: [{ required: true, trigger: 'blur', validator: validateUsername }],
        password: [{ required: true, trigger: 'blur', validator: validatePassword }]
      },
      passwordType: 'password',
      capsTooltip: false,
      loading: false,
      configElLoading: true,
      apiBaseUrl: 'http://10.134.13.242:5005',
      apiPrefix: '/api'
    }
  },
  created() {
    console.log('登录组件已创建')
    this.clearAuthTokens()
    if (this.$route.query.redirect) {
      history.replaceState(null, null, this.$route.path)
    }
    this.loadConfig()
  },
  methods: {
    async loadConfig() {
      try {
        console.log('开始加载配置文件')
        const response = await configApi.get('/config.ini', { responseType: 'text' })
        const configContent = response.data
        const lines = configContent.split('\n')
        let inApiSection = false
        for (const line of lines) {
          const trimmedLine = line.trim()
          if (trimmedLine === '[api]') {
            inApiSection = true
            continue
          }
          if (inApiSection && trimmedLine.startsWith('prefix')) {
            const parts = trimmedLine.split('=')
            if (parts.length >= 2) {
              this.apiPrefix = parts[1].trim()
              console.log('接口前缀:', this.apiPrefix)
            }
          }
          if (inApiSection && trimmedLine.startsWith('baseUrl')) {
            const parts = trimmedLine.split('=')
            if (parts.length >= 2) {
              this.apiBaseUrl = parts[1].trim()
              console.log('API地址:', this.apiBaseUrl)
              break
            }
          }
          if (inApiSection && trimmedLine.startsWith('[')) break
        }
      } catch (error) {
        console.warn('配置加载失败，用兜底地址:', error.message)
      } finally {
        this.configElLoading = false
      }
    },
    checkCapslock(e) {
      const { key } = e
      this.capsTooltip = key && key.length === 1 && (key >= 'A' && key <= 'Z')
    },
    showPwd() {
      this.passwordType = this.passwordType === 'password' ? '' : 'password'
      this.$nextTick(() => this.$refs.password.focus())
    },
    clearAuthTokens() {
      localStorage.removeItem('token')
      sessionStorage.removeItem('token')
      store.commit('user/SET_TOKEN', '')
      // 重置路由：低版本Vue Router不支持removeRoute，直接替换routes
      router.options.routes = router.options.routes.filter(route =>
        !asyncRoutes.some(asyncRoute => asyncRoute.path === route.path)
      )
      // Vue Router 4 不需要手动绑定match方法
    },
    handleLogin() {
      if (this.configElLoading) {
        this.$message.warning('配置加载中，请稍候')
        return
      }
      this.$refs.loginForm.validate(valid => {
        if (!valid) return false
        this.loading = true
        const loginUrl = `${this.apiBaseUrl}${this.apiPrefix}/login`
        console.log('最终登录地址:', loginUrl)

        axios.post(loginUrl, this.loginForm)
          .then(response => {
            const token = response.data.token || ''
            if (token) {
              // 存储Token
              localStorage.setItem('token', token)
              sessionStorage.setItem('token', token)
              store.commit('user/SET_TOKEN', token)

              // 关键修复：Vue Router 4 使用 addRoute 方法添加路由
              // 过滤已存在的路由，避免重复添加
              const newRoutes = asyncRoutes.filter(route =>
                !router.options.routes.some(r => r.path === route.path)
              )
              // Vue Router 4 使用 addRoute 方法，需要遍历添加
              newRoutes.forEach(route => {
                router.addRoute(route)
              })
              router.options.routes = [...router.options.routes, ...newRoutes] // 更新路由配置

              this.$message.success('登录成功，马上跳转！')
              // 直接跳转到样本页，强制刷新页面
              setTimeout(() => {
                window.location.href = '#/sample'
                window.location.reload() // 确保路由和侧边栏刷新
              }, 300)
            } else {
              this.loading = false
              this.$message.error('登录失败：未获取到Token')
            }
          })
          .catch(error => {
            console.error('登录错误:', error)
            this.loading = false
            let errMsg = '登录失败：'
            if (error.code === 'ERR_NETWORK') errMsg += '网络不通'
            else if (error.response?.status === 404) errMsg += '接口不存在'
            else if (error.response?.status === 401) errMsg += '用户名或密码错误'
            else errMsg += error.message
            this.$message.error(errMsg)
          })
      })
    }
  }
}
</script>

<style lang="scss">
$bg-gradient: linear-gradient(to bottom right, #2a3747, #354459);
$form-bg: rgba(45, 58, 75, 0.8);
$primary-color: #409EFF;
$border-color: rgba(255, 255, 255, 0.1);
$input-focus: rgba(64, 158, 255, 0.2);

.login-container {
  min-height: 100vh;
  width: 100%;
  background: $bg-gradient;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
  box-sizing: border-box;
}

.login-form {
  position: relative;
  width: 520px;
  max-width: 100%;
  padding: 80px 35px 40px;
  background: $form-bg;
  border-radius: 10px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(10px);
}

.title-container {
  text-align: center;
  margin-bottom: 40px;
  position: relative;

  .title {
    font-size: 28px;
    color: #fff;
    margin: 0 0 10px 0;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  }

  &:after {
    content: '';
    position: absolute;
    bottom: -20px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background: $primary-color;
    border-radius: 3px;
  }
}

.el-form-item {
  border: 1px solid $border-color;
  background: rgba(0, 0, 0, 0.1);
  border-radius: 5px;
  margin-bottom: 20px;
  transition: all 0.3s ease;

  &:focus, &:hover {
    border-color: $primary-color;
    box-shadow: 0 0 0 2px $input-focus;
  }
}

.svg-container {
  padding: 6px 5px 6px 15px;
  color: #889aa4;
  vertical-align: middle;
  width: 30px;
  display: inline-block;
}

.el-input {
  width: 100%;

  input {
    background: transparent;
    border: 0;
    -webkit-appearance: none;
    border-radius: 0;
    padding: 12px 5px 12px 15px;
    color: #fff;
    height: 47px;
    caret-color: #fff;
  }
}

.show-pwd {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 18px;
  color: #889aa4;
  cursor: pointer;
  user-select: none;
}

.el-button {
  &.el-button--primary {
    background: $primary-color;
    border-color: $primary-color;
    transition: all 0.3s ease;
    font-weight: 500;

    &:hover {
      background: #66b1ff;
      border-color: #66b1ff;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(64, 158, 255, 0.5);
    }

    &:disabled {
      background: $primary-color;
      border-color: $primary-color;
      opacity: 0.7;
      cursor: not-allowed;
    }
  }
}

@media only screen and (max-width: 576px) {
  .login-form {
    padding: 60px 20px 30px;
    width: 100%;
  }
}
</style>
