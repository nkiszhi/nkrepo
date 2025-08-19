<template>
  <div class="register-container">
    <el-form ref="registerForm" :model="registerForm" :rules="registerRules" class="register-form" autocomplete="on" label-position="left">
      <div class="title-container">
        <h4 class="title">基于可信度评估的多模型恶意代码检测系统</h4>
        <h4 class="title">注册账号</h4>
      </div>
      
      <el-form-item prop="username">
        <span class="svg-container">
          <svg-icon icon-class="user" />
        </span>
        <el-input
          v-model="registerForm.username"
          placeholder="用户名"
          name="username"
          type="text"
          tabindex="1"
          autocomplete="on"
        />
      </el-form-item>
      
      <el-form-item prop="email">
        <span class="svg-container">
          <svg-icon icon-class="email" />
        </span>
        <el-input
          v-model="registerForm.email"
          placeholder="电子邮箱"
          name="email"
          type="email"
          tabindex="2"
          autocomplete="on"
        />
      </el-form-item>
      
      <el-form-item prop="invitationCode">
        <span class="svg-container">
          <svg-icon icon-class="key" />
        </span>
        <el-input
          v-model="registerForm.invitationCode"
          placeholder="邀请码"
          name="invitationCode"
          type="text"
          tabindex="3"
          autocomplete="off"
        />
      </el-form-item>
      
      <el-tooltip v-model="capsTooltip" content="大写锁定已开启" placement="right" manual>
        <el-form-item prop="password">
          <span class="svg-container">
            <svg-icon icon-class="password" />
          </span>
          <el-input
            :key="passwordType"
            v-model="registerForm.password"
            :type="passwordType"
            placeholder="密码"
            name="password"
            tabindex="4"
            autocomplete="on"
            @keyup.native="checkCapslock"
            @blur="capsTooltip = false"
          />
          <span class="show-pwd" @click="togglePasswordVisibility">
            <svg-icon :icon-class="passwordType === 'password' ? 'eye' : 'eye-open'" />
          </span>
        </el-form-item>
      </el-tooltip>
      
      <el-form-item prop="confirmPassword">
        <span class="svg-container">
          <svg-icon icon-class="password" />
        </span>
        <el-input
          :key="confirmPasswordType"
          v-model="registerForm.confirmPassword"
          :type="confirmPasswordType"
          placeholder="确认密码"
          name="confirmPassword"
          tabindex="5"
          autocomplete="on"
        />
        <span class="show-pwd" @click="toggleConfirmPasswordVisibility">
          <svg-icon :icon-class="confirmPasswordType === 'password' ? 'eye' : 'eye-open'" />
        </span>
      </el-form-item>
      
      <el-button :loading="loading" type="primary" style="width:100%;margin-bottom:15px;" @click.native.prevent="handleRegister">
        注册
      </el-button>
      
      <el-button 
        type="text" 
        style="color: #409EFF; text-decoration: underline; width: 100%; margin-top: 10px;"
        @click="goToLogin"
      >
        已有账号？立即登录
      </el-button>
    </el-form>
    
    <el-dialog v-model="invitationErrorVisible" title="邀请码错误" width="30%">
      <div slot="content">
        <p>您输入的邀请码无效或已被使用，请检查后重新输入。</p>
      </div>
      <div slot="footer" class="dialog-footer">
        <el-button @click="invitationErrorVisible = false">关闭</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import { validEmail } from '@/utils/validate'
import router from '@/router'
import store from '@/store'

export default {
  name: 'Register',
  data() {
    const validateConfirmPassword = (rule, value, callback) => {
      if (value !== this.registerForm.password) {
        callback(new Error('两次输入的密码不一致'))
      } else {
        callback()
      }
    }
    return {
      registerForm: {
        username: '',
        email: '',
        invitationCode: '',
        password: '',
        confirmPassword: ''
      },
      registerRules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' },
          { min: 3, max: 20, message: '用户名长度在3到20个字符', trigger: 'blur' }
        ],
        email: [
          { required: true, message: '请输入电子邮箱', trigger: 'blur' },
          { validator: validEmail, trigger: 'blur' }
        ],
        invitationCode: [
          { required: true, message: '请输入邀请码', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' },
          { min: 6, message: '密码长度不能小于6位', trigger: 'blur' }
        ],
        confirmPassword: [
          { required: true, message: '请确认密码', trigger: 'blur' },
          { validator: validateConfirmPassword, trigger: 'blur' }
        ]
      },
      passwordType: 'password',
      confirmPasswordType: 'password',
      capsTooltip: false,
      loading: false,
      invitationErrorVisible: false
    }
  },
  created() {
    console.log('注册组件已创建')
    this.clearAuthTokens()
  },
  methods: {
    togglePasswordVisibility() {
      this.passwordType = this.passwordType === 'password' ? '' : 'password'
    },
    toggleConfirmPasswordVisibility() {
      this.confirmPasswordType = this.confirmPasswordType === 'password' ? '' : 'password'
    },
    checkCapslock(e) {
      const { key } = e
      this.capsTooltip = key && key.length === 1 && (key >= 'A' && key <= 'Z')
    },
    clearAuthTokens() {
      localStorage.removeItem('token')
      sessionStorage.removeItem('token')
      store.commit('user/SET_TOKEN', '')
      console.log('所有认证token已清除')
    },
    handleRegister() {
      console.log('开始注册流程')
      this.$refs.registerForm.validate(valid => {
        if (valid) {
          this.loading = true
          console.log('表单验证通过，准备发送注册请求')
          
          // 模拟注册请求，实际项目中请替换为真实API调用
          store.dispatch('user/register', this.registerForm)
            .then(response => {
              console.log('注册成功，响应数据:', response)
              this.loading = false
              this.$message({
                type: 'success',
                message: '注册成功，请使用新账号登录',
                duration: 1500,
                onClose: () => {
                  console.log('导航到登录页')
                  router.push('/login')
                }
              })
            })
            .catch(error => {
              console.error('注册失败:', error)
              this.loading = false
              if (error.response && error.response.data && error.response.data.message) {
                if (error.response.data.message.includes('邀请码')) {
                  this.invitationErrorVisible = true
                } else {
                  this.$message.error(error.response.data.message)
                }
              } else {
                this.$message.error('注册失败，请稍后再试')
              }
            })
        } else {
          console.log('表单验证失败')
          return false
        }
      })
    },
    goToLogin() {
      console.log('返回登录页按钮被点击')
      this.clearAuthTokens()
      router.push('/login')
    }
  }
}
</script>

<style lang="scss" scoped>
$bg-gradient: linear-gradient(to bottom right, #2a3747, #354459);
$form-bg: rgba(45, 58, 75, 0.8);
$primary-color: #409EFF;
$border-color: rgba(255, 255, 255, 0.1);

.register-container {
  min-height: 100vh;
  width: 100%;
  background: $bg-gradient;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
  box-sizing: border-box;
}

.register-form {
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
  
  &.el-button--text {
    color: $primary-color;
    font-weight: 500;
  }
}

@media only screen and (max-width: 576px) {
  .register-form {
    padding: 60px 20px 30px;
    width: 100%;
  }
}
</style>