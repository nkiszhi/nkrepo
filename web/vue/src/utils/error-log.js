import { nextTick } from 'vue'
import { useErrorLogStore } from '@/stores/errorLog'
import { isString, isArray } from '@/utils/validate'
import settings from '@/settings'

// you can set in settings.js
// errorLog:'production' | ['production', 'development']
const { errorLog: needErrorLog } = settings

function checkNeed() {
  const env = process.env.NODE_ENV
  if (isString(needErrorLog)) {
    return env === needErrorLog
  }
  if (isArray(needErrorLog)) {
    return needErrorLog.includes(env)
  }
  return false
}

export function setupErrorLog(app) {
  if (checkNeed()) {
    app.config.errorHandler = function(err, vm, info) {
      // Don't ask me why I use nextTick, it just a hack.
      // detail see https://forum.vuejs.org/t/dispatch-in-vue-config-errorhandler-has-some-problem/23500
      nextTick(() => {
        const errorLogStore = useErrorLogStore()
        errorLogStore.addErrorLog({
          err,
          vm,
          info,
          url: window.location.href
        })
        console.error(err, info)
      })
    }
  }
}
