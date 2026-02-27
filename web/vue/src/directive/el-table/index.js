import adaptive from './adaptive.js'

const install = function(Vue) {
  Vue.directive('el-height-adaptive-table', adaptive)
}

if (window.Vue) {
  window['el-height-adaptive-table'] = adaptive
  Vue.use(install); // eslint-disable-line
}

adaptive.install = install
export default adaptive
