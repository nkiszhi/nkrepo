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
      
      <el-button :loading="loading" type="primary" style="width:100%;margin-bottom:30px;" @click.native.prevent="handleLogin">
        登录
      </el-button>
    </el-form>
  </div>
</template>

<script>
import { validUsername } from '@/utils/validate'
import router from '@/router'
import store from '@/store'

export default {
  name: 'Login',
  data() {
    const validateUsername = (rule, value, callback) => {
      if (!validUsername(value)) {
        callback(new Error('请输入正确的用户名'))
      } else {
        callback()
      }
    }
    const validatePassword = (rule, value, callback) => {
      if (value.length < 6) {
        callback(new Error('密码长度不能小于6位'))
      } else {
        callback()
      }
    }
    return {
      loginForm: {
        username: 'admin',
        password: '111111'
      },
      loginRules: {
        username: [{ required: true, trigger: 'blur', validator: validateUsername }],
        password: [{ required: true, trigger: 'blur', validator: validatePassword }]
      },
      passwordType: 'password',
      capsTooltip: false,
      loading: false
    }
  },
  created() {
    console.log('登录组件已创建')
    this.clearAuthTokens()
    
    // 清除URL中的redirect参数
    if (this.$route.query.redirect) {
      console.log('清除URL中的redirect参数')
      history.replaceState(null, null, this.$route.path)
    }
  },
  methods: {
    checkCapslock(e) {
      const { key } = e
      this.capsTooltip = key && key.length === 1 && (key >= 'A' && key <= 'Z')
    },
    showPwd() {
      this.passwordType = this.passwordType === 'password' ? '' : 'password'
      this.$nextTick(() => {
        this.$refs.password.focus()
      })
    },
    clearAuthTokens() {
      localStorage.removeItem('token')
      sessionStorage.removeItem('token')
      store.commit('user/SET_TOKEN', '')
      console.log('所有认证token已清除')
    },
    handleLogin() {
      this.$refs.loginForm.validate(valid => {
        if (valid) {
          this.loading = true
          console.log('开始登录请求，用户名:', this.loginForm.username)
          
          store.dispatch('user/login', this.loginForm)
            .then(() => {
              console.log('登录成功')
              router.push('/')
              this.loading = false
            })
            .catch(error => {
              console.error('登录失败:', error)
              this.loading = false
              this.$message.error('登录失败，请检查用户名和密码')
            })
        } else {
          console.log('表单验证失败')
          return false
        }
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
  }
}

@media only screen and (max-width: 576px) {
  .login-form {
    padding: 60px 20px 30px;
    width: 100%;
  }
}
</style>