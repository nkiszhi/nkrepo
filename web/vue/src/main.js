import { createApp } from 'vue'

import Cookies from 'js-cookie'

import 'normalize.css/normalize.css' // a modern alternative to CSS resets

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'

import '@/styles/index.scss' // global css

import App from './App'
import pinia from './stores'
import router from './router'

import icons from './icons' // icon
import './permission' // permission control
import { setupErrorLog } from './utils/error-log' // error log

import * as filters from './filters' // global filters

const app = createApp(App)

app.use(pinia)
app.use(router)
app.use(ElementPlus, {
  size: Cookies.get('size') || 'default',
  locale: zhCn
})
app.use(icons)

// register global utility filters as methods
app.config.globalProperties.$filters = filters

// Setup error logging
setupErrorLog(app)

app.mount('#app')

// Export app for external access if needed
export { app }
