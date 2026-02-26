<template>
  <div ref="rightPanel" :class="{show:show}" class="rightPanel-container">
    <div class="rightPanel-background" />
    <div class="rightPanel">
      <div class="handle-button" :style="{'top':buttonTop+'px','background-color':theme}" @click="show=!show">
        <el-icon :class="show?'el-icon-close':'el-icon-setting'">
          <component :is="show ? 'Close' : 'Setting'" />
        </el-icon>
      </div>
      <div class="rightPanel-items">
        <slot />
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import { Close, Setting } from '@element-plus/icons-vue'
import { addClass, removeClass } from '@/utils'
import { useSettingsStore } from '@/stores/settings'

export default {
  name: 'RightPanel',
  components: {
    Close,
    Setting
  },
  props: {
    clickNotClose: {
      default: false,
      type: Boolean
    },
    buttonTop: {
      default: 250,
      type: Number
    }
  },
  setup(props) {
    const settingsStore = useSettingsStore()
    const rightPanel = ref(null)
    const show = ref(false)

    const theme = computed(() => settingsStore.theme)

    const addEventClick = () => {
      window.addEventListener('click', closeSidebar)
    }

    const closeSidebar = (evt) => {
      const parent = evt.target.closest('.rightPanel')
      if (!parent) {
        show.value = false
        window.removeEventListener('click', closeSidebar)
      }
    }

    const insertToBody = () => {
      const elx = rightPanel.value
      const body = document.querySelector('body')
      body.insertBefore(elx, body.firstChild)
    }

    watch(show, (value) => {
      if (value && !props.clickNotClose) {
        addEventClick()
      }
      if (value) {
        addClass(document.body, 'showRightPanel')
      } else {
        removeClass(document.body, 'showRightPanel')
      }
    })

    onMounted(() => {
      insertToBody()
    })

    onBeforeUnmount(() => {
      const elx = rightPanel.value
      if (elx) {
        elx.remove()
      }
    })

    return {
      rightPanel,
      show,
      theme
    }
  }
}
</script>

<style>
.showRightPanel {
  overflow: hidden;
  position: relative;
  width: calc(100% - 15px);
}
</style>

<style lang="scss" scoped>
.rightPanel-background {
  position: fixed;
  top: 0;
  left: 0;
  opacity: 0;
  transition: opacity .3s cubic-bezier(.7, .3, .1, 1);
  background: rgba(0, 0, 0, .2);
  z-index: -1;
}

.rightPanel {
  width: 100%;
  max-width: 260px;
  height: 100vh;
  position: fixed;
  top: 0;
  right: 0;
  box-shadow: 0px 0px 15px 0px rgba(0, 0, 0, .05);
  transition: all .25s cubic-bezier(.7, .3, .1, 1);
  transform: translate(100%);
  background: #fff;
  z-index: 40000;
}

.show {
  transition: all .3s cubic-bezier(.7, .3, .1, 1);

  .rightPanel-background {
    z-index: 20000;
    opacity: 1;
    width: 100%;
    height: 100%;
  }

  .rightPanel {
    transform: translate(0);
  }
}

.handle-button {
  width: 48px;
  height: 48px;
  position: absolute;
  left: -48px;
  text-align: center;
  font-size: 24px;
  border-radius: 6px 0 0 6px !important;
  z-index: 0;
  pointer-events: auto;
  cursor: pointer;
  color: #fff;
  line-height: 48px;
  i {
    font-size: 24px;
    line-height: 48px;
  }
}
</style>
