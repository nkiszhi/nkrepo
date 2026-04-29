<template>
  <div class="login-container">
    <!-- 动态背景 -->
    <div class="bg-animation">
      <div class="wave wave1"></div>
      <div class="wave wave2"></div>
      <div class="wave wave3"></div>
    </div>
    
    <!-- 装饰元素 -->
    <div class="decoration">
      <div class="circle circle1"></div>
      <div class="circle circle2"></div>
      <div class="circle circle3"></div>
    </div>

    <el-form ref="loginForm" :model="loginForm" :rules="loginRules" class="login-form" autocomplete="on" label-position="left">
      <!-- Logo 和标题区域 -->
      <div class="title-container">
        <div class="logo-wrapper">
          <div class="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
        </div>
        <h3 class="title">恶意代码检测系统</h3>
        <p class="subtitle">Malware Detection System</p>
        <div class="title-line"></div>
      </div>

      <!-- 用户名输入 -->
      <el-form-item prop="username" class="form-item-wrapper">
        <div class="input-wrapper" :class="{ 'focused': usernameFocused }">
          <span class="svg-container">
            <svg-icon icon-class="user" />
          </span>
          <el-input
            ref="username"
            v-model="loginForm.username"
            placeholder="请输入用户名"
            name="username"
            type="text"
            tabindex="1"
            autocomplete="on"
            @focus="usernameFocused = true"
            @blur="usernameFocused = false"
          />
        </div>
      </el-form-item>

      <!-- 密码输入 -->
      <el-tooltip v-model="capsTooltip" content="大写锁定已开启" placement="right" manual>
        <el-form-item prop="password" class="form-item-wrapper">
          <div class="input-wrapper" :class="{ 'focused': passwordFocused }">
            <span class="svg-container">
              <svg-icon icon-class="password" />
            </span>
            <el-input
              :key="passwordType"
              ref="password"
              v-model="loginForm.password"
              :type="passwordType"
              placeholder="请输入密码"
              name="password"
              tabindex="2"
              autocomplete="on"
              @focus="passwordFocused = true"
              @blur="passwordFocused = false"
              @keyup.native="checkCapslock"
              @keyup.enter.native="handleLogin"
            />
            <span class="show-pwd" @click="showPwd">
              <svg-icon :icon-class="passwordType === 'password' ? 'eye' : 'eye-open'" />
            </span>
          </div>
        </el-form-item>
      </el-tooltip>

      <!-- 登录按钮 -->
      <el-button 
        :loading="loading || configElLoading" 
        type="primary" 
        class="login-button"
        :disabled="configElLoading" 
        @click.native.prevent="handleLogin"
      >
        <span v-if="configElLoading">加载配置中...</span>
        <span v-else>登 录</span>
      </el-button>

      <!-- 底部信息 -->
      <div class="footer-info">
        <p>南开大学反病毒实验室 NKAMG</p>
      </div>
    </el-form>
  </div>
</template>

<script>
import axios from 'axios'
import router from '@/router'
import store from '@/store'
import { asyncRoutes } from '@/router'

const configApi = axios.create({
  timeout: 5000
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
      apiPrefix: '/api',
      usernameFocused: false,
      passwordFocused: false
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
      router.options.routes = router.options.routes.filter(route =>
        !asyncRoutes.some(asyncRoute => asyncRoute.path === route.path)
      )
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
              localStorage.setItem('token', token)
              sessionStorage.setItem('token', token)
              store.commit('user/SET_TOKEN', token)

              const newRoutes = asyncRoutes.filter(route =>
                !router.options.routes.some(r => r.path === route.path)
              )
              newRoutes.forEach(route => {
                router.addRoute(route)
              })
              router.options.routes = [...router.options.routes, ...newRoutes]

              this.$message.success('登录成功，马上跳转！')
              setTimeout(() => {
                window.location.href = '#/sample'
                window.location.reload()
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
.login-container {
  min-height: 100vh;
  width: 100%;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

// 动态波浪背景
.bg-animation {
  position: absolute;
  width: 100%;
  height: 100%;
  overflow: hidden;
  z-index: 0;
}

.wave {
  position: absolute;
  width: 200%;
  height: 200%;
  background: radial-gradient(ellipse at center, rgba(64, 158, 255, 0.1) 0%, transparent 70%);
  animation: wave-animation 15s ease-in-out infinite;
}

.wave1 {
  top: -50%;
  left: -50%;
  animation-delay: 0s;
}

.wave2 {
  top: -30%;
  left: -30%;
  animation-delay: -5s;
  opacity: 0.5;
}

.wave3 {
  top: -70%;
  left: -70%;
  animation-delay: -10s;
  opacity: 0.3;
}

@keyframes wave-animation {
  0%, 100% { transform: rotate(0deg) scale(1); }
  50% { transform: rotate(180deg) scale(1.2); }
}

// 装饰圆形
.decoration {
  position: absolute;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
}

.circle {
  position: absolute;
  border-radius: 50%;
  background: linear-gradient(135deg, rgba(64, 158, 255, 0.2), rgba(103, 194, 58, 0.1));
  animation: float 6s ease-in-out infinite;
}

.circle1 {
  width: 300px;
  height: 300px;
  top: -100px;
  right: -100px;
  animation-delay: 0s;
}

.circle2 {
  width: 200px;
  height: 200px;
  bottom: -50px;
  left: -50px;
  animation-delay: -2s;
}

.circle3 {
  width: 150px;
  height: 150px;
  top: 50%;
  left: 10%;
  animation-delay: -4s;
}

@keyframes float {
  0%, 100% { transform: translateY(0) rotate(0deg); }
  50% { transform: translateY(-20px) rotate(10deg); }
}

// 登录表单
.login-form {
  position: relative;
  z-index: 1;
  width: 480px;
  max-width: 100%;
  padding: 50px 40px 40px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 20px;
  box-shadow: 
    0 25px 50px rgba(0, 0, 0, 0.3),
    0 0 0 1px rgba(255, 255, 255, 0.1),
    inset 0 0 80px rgba(255, 255, 255, 0.02);
  backdrop-filter: blur(20px);
  animation: form-appear 0.6s ease-out;
}

@keyframes form-appear {
  from {
    opacity: 0;
    transform: translateY(30px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

// Logo 和标题
.title-container {
  text-align: center;
  margin-bottom: 40px;
}

.logo-wrapper {
  margin-bottom: 20px;
}

.logo-icon {
  width: 60px;
  height: 60px;
  margin: 0 auto;
  color: #409EFF;
  animation: logo-glow 2s ease-in-out infinite;
  
  svg {
    width: 100%;
    height: 100%;
  }
}

@keyframes logo-glow {
  0%, 100% { 
    filter: drop-shadow(0 0 5px rgba(64, 158, 255, 0.5));
    transform: scale(1);
  }
  50% { 
    filter: drop-shadow(0 0 20px rgba(64, 158, 255, 0.8));
    transform: scale(1.05);
  }
}

.title {
  font-size: 26px;
  color: #fff;
  margin: 0 0 8px 0;
  font-weight: 600;
  letter-spacing: 2px;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.subtitle {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.6);
  margin: 0 0 20px 0;
  letter-spacing: 1px;
  font-weight: 300;
}

.title-line {
  width: 80px;
  height: 3px;
  background: linear-gradient(90deg, transparent, #409EFF, transparent);
  margin: 0 auto;
  border-radius: 3px;
}

// 表单项
.form-item-wrapper {
  margin-bottom: 25px;
  
  .el-form-item__error {
    padding-top: 8px;
    color: #f56c6c;
    font-size: 12px;
  }
}

.input-wrapper {
  display: flex;
  align-items: center;
  width: 100%;
  height: 50px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 12px;
  transition: all 0.3s ease;
  overflow: hidden;
  
  &:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(64, 158, 255, 0.5);
  }
  
  &.focused {
    background: rgba(255, 255, 255, 0.15);
    border-color: #409EFF;
    box-shadow: 
      0 0 0 3px rgba(64, 158, 255, 0.2),
      0 0 20px rgba(64, 158, 255, 0.1);
    
    .svg-container {
      color: #409EFF;
    }
  }
}

.svg-container {
  padding: 0 15px;
  color: rgba(255, 255, 255, 0.5);
  transition: color 0.3s ease;
  
  .svg-icon {
    width: 20px;
    height: 20px;
  }
}

.el-input {
  flex: 1;
  
  .el-input__wrapper {
    background: transparent;
    box-shadow: none;
    padding: 15px 0;
    
    .el-input__inner {
      color: #fff;
      font-size: 15px;
      letter-spacing: 0.5px;
      
      &::placeholder {
        color: rgba(255, 255, 255, 0.4);
      }
    }
  }
}

.show-pwd {
  padding: 0 15px;
  color: rgba(255, 255, 255, 0.5);
  cursor: pointer;
  transition: color 0.3s ease;
  
  &:hover {
    color: #409EFF;
  }
  
  .svg-icon {
    width: 18px;
    height: 18px;
  }
}

// 登录按钮
.login-button {
  width: 100%;
  height: 50px;
  margin-top: 10px;
  margin-bottom: 30px;
  font-size: 16px;
  font-weight: 500;
  letter-spacing: 3px;
  border-radius: 12px;
  background: linear-gradient(135deg, #409EFF 0%, #66b1ff 100%);
  border: none;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
  }
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: 
      0 10px 30px rgba(64, 158, 255, 0.4),
      0 0 0 1px rgba(255, 255, 255, 0.2);
    
    &::before {
      left: 100%;
    }
  }
  
  &:active {
    transform: translateY(-1px);
  }
  
  &.is-disabled {
    background: linear-gradient(135deg, #409EFF 0%, #66b1ff 100%);
    opacity: 0.7;
  }
}

// 底部信息
.footer-info {
  text-align: center;
  
  p {
    margin: 0;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.4);
    letter-spacing: 1px;
  }
}

// 响应式设计
@media only screen and (max-width: 576px) {
  .login-form {
    padding: 40px 25px 30px;
    width: 100%;
    border-radius: 15px;
  }
  
  .title {
    font-size: 22px;
  }
  
  .logo-icon {
    width: 50px;
    height: 50px;
  }
  
  .circle1, .circle2, .circle3 {
    display: none;
  }
}
</style>
