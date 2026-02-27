import { createStore } from 'vuex'
import getters from './getters.js'

const modulesFiles = import.meta.glob('./modules/*.js', { eager: true })

const modules = {}
for (const path in modulesFiles) {
  const moduleName = path.replace(/^\.\/modules\/(.*)\.\w+$/, '$1')
  modules[moduleName] = modulesFiles[path].default
}

const store = createStore({
  modules,
  getters
})

export default store
