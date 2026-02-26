import adaptive from './adaptive'

const install = function(app) {
  app.directive('el-height-adaptive-table', adaptive)
}

adaptive.install = install
export default adaptive
