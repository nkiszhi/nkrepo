<template>
  <div :class="{'has-logo':showLogo}">
    <logo v-if="showLogo" :collapse="isCollapse" />
    <el-scrollbar wrap-class="scrollbar-wrapper">
      <el-menu
        :default-active="activeMenu"
        :collapse="isCollapse"
        :background-color="variables.menuBg"
        :text-color="variables.menuText"
        :unique-opened="false"
        :active-text-color="variables.menuActiveText"
        :collapse-transition="false"
        mode="vertical"
      >
        <sidebar-item v-for="route in permission_routes" :key="route.path" :item="route" :base-path="route.path" />
      </el-menu>
    </el-scrollbar>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import Logo from './Logo'
import SidebarItem from './SidebarItem'
import variables from '@/styles/variables.scss'
import { usePermissionStore } from '@/stores/permission'
import { useAppStore } from '@/stores/app'
import { useSettingsStore } from '@/stores/settings'

export default {
  components: { SidebarItem, Logo },
  setup() {
    const route = useRoute()
    const permissionStore = usePermissionStore()
    const appStore = useAppStore()
    const settingsStore = useSettingsStore()

    const permission_routes = computed(() => permissionStore.routes)
    const sidebar = computed(() => appStore.sidebar)

    const activeMenu = computed(() => {
      const { meta, path } = route
      // if set path, the sidebar will highlight the path you set
      if (meta.activeMenu) {
        return meta.activeMenu
      }
      return path
    })

    const showLogo = computed(() => settingsStore.sidebarLogo)

    const isCollapse = computed(() => !sidebar.value.opened)

    return {
      permission_routes,
      activeMenu,
      showLogo,
      variables,
      isCollapse
    }
  }
}
</script>
