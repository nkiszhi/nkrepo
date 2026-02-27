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
        <!-- 遍历permission_routes（动态生成的路由），渲染侧边栏 -->
        <sidebar-item v-for="route in permission_routes" :key="route.path || route.name" :item="route" :base-path="route.path" />
      </el-menu>
    </el-scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import Logo from './Logo.vue'
import SidebarItem from './SidebarItem.vue'

export default {
  components: { SidebarItem, Logo },
  computed: {
    ...mapGetters([
      'permission_routes', // 从Vuex获取动态路由（已包含常量路由+异步路由）
      'sidebar'
    ]),
    activeMenu() {
      const route = this.$route
      const { meta, path } = route
      // 优先使用meta中配置的activeMenu，否则用当前路径
      if (meta && meta.activeMenu) {
        return meta.activeMenu
      }
      return path
    },
    showLogo() {
      // 从settings中获取是否显示Logo的配置
      return this.$store.state.settings.sidebarLogo
    },
    variables() {
      // 直接定义SCSS变量,避免Vite中导入SCSS文件的问题
      return {
        menuText: '#bfcbd9',
        menuActiveText: '#409EFF',
        subMenuActiveText: '#f4f4f5',
        menuBg: '#304156',
        menuHover: '#263445',
        subMenuBg: '#1f2d3d',
        subMenuHover: '#001528',
        sideBarWidth: '210px'
      }
    },
    isCollapse() {
      // 侧边栏折叠状态（取反，opened为true时不折叠）
      return !this.sidebar.opened
    }
  },
  watch: {
    // 监听路由变化，强制刷新侧边栏（确保动态路由添加后渲染）
    $route() {
      this.$forceUpdate()
    },
    // 监听permission_routes变化，刷新侧边栏
    permission_routes() {
      this.$forceUpdate()
    }
  }
}
</script>

<style scoped>
.scrollbar-wrapper {
  height: calc(100vh - 48px);
  overflow-y: auto;
}

.has-logo {
  padding-top: 20px;
}
</style>
