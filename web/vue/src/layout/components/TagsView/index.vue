<template>
  <div id="tags-view-container" class="tags-view-container">
    <scroll-pane ref="scrollPane" class="tags-view-wrapper" @scroll="handleScroll">
      <router-link
        v-for="tag in visitedViews"
        ref="tagRefs"
        :key="tag.path"
        :class="isActive(tag)?'active':''"
        :to="{ path: tag.path, query: tag.query, fullPath: tag.fullPath }"
        custom
        v-slot="{ navigate }"
      >
        <span
          class="tags-view-item"
          @click="navigate"
          @click.middle="!isAffix(tag)?closeSelectedTag(tag):''"
          @contextmenu.prevent="openMenu(tag,$event)"
        >
          {{ tag.title }}
          <span v-if="!isAffix(tag)" class="el-icon-close" @click.prevent.stop="closeSelectedTag(tag)" />
        </span>
      </router-link>
    </scroll-pane>
    <ul v-show="visible" :style="{left:left+'px',top:top+'px'}" class="contextmenu">
      <li @click="refreshSelectedTag(selectedTag)">Refresh</li>
      <li v-if="!isAffix(selectedTag)" @click="closeSelectedTag(selectedTag)">Close</li>
      <li @click="closeOthersTags">Close Others</li>
      <li @click="closeAllTags(selectedTag)">Close All</li>
    </ul>
  </div>
</template>

<script>
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import ScrollPane from './ScrollPane'
import path from 'path'
import { useTagsViewStore } from '@/stores/tagsView'
import { usePermissionStore } from '@/stores/permission'

export default {
  components: { ScrollPane },
  setup() {
    const route = useRoute()
    const router = useRouter()
    const tagsViewStore = useTagsViewStore()
    const permissionStore = usePermissionStore()

    const visible = ref(false)
    const top = ref(0)
    const left = ref(0)
    const selectedTag = ref({})
    const affixTags = ref([])
    const scrollPane = ref(null)
    const tagRefs = ref([])

    const visitedViews = computed(() => tagsViewStore.visitedViews)
    const routes = computed(() => permissionStore.routes)

    const isActive = (routeItem) => {
      return routeItem.path === route.path
    }

    const isAffix = (tag) => {
      return tag.meta && tag.meta.affix
    }

    const filterAffixTags = (routesList, basePath = '/') => {
      let tags = []
      routesList.forEach(routeItem => {
        if (routeItem.meta && routeItem.meta.affix) {
          const tagPath = path.resolve(basePath, routeItem.path)
          tags.push({
            fullPath: tagPath,
            path: tagPath,
            name: routeItem.name,
            meta: { ...routeItem.meta }
          })
        }
        if (routeItem.children) {
          const tempTags = filterAffixTags(routeItem.children, routeItem.path)
          if (tempTags.length >= 1) {
            tags = [...tags, ...tempTags]
          }
        }
      })
      return tags
    }

    const initTags = () => {
      affixTags.value = filterAffixTags(routes.value)
      for (const tag of affixTags.value) {
        // Must have tag name
        if (tag.name) {
          tagsViewStore.addVisitedView(tag)
        }
      }
    }

    const addTags = () => {
      const { name } = route
      if (name) {
        tagsViewStore.addView(route)
      }
      return false
    }

    const moveToCurrentTag = () => {
      nextTick(() => {
        const tags = tagRefs.value
        if (tags && tags.length) {
          for (const tag of tags) {
            if (tag && tag.$el && tag.to && tag.to.path === route.path) {
              scrollPane.value?.moveToTarget(tag)
              // when query is different then update
              if (tag.to.fullPath !== route.fullPath) {
                tagsViewStore.updateVisitedView(route)
              }
              break
            }
          }
        }
      })
    }

    const refreshSelectedTag = (view) => {
      tagsViewStore.delCachedView(view)
      const { fullPath } = view
      nextTick(() => {
        router.replace({
          path: '/redirect' + fullPath
        })
      })
    }

    const closeSelectedTag = (view) => {
      const { visitedViews } = tagsViewStore.delView(view)
      if (isActive(view)) {
        toLastView(visitedViews, view)
      }
    }

    const closeOthersTags = () => {
      router.push(selectedTag.value)
      tagsViewStore.delOthersViews(selectedTag.value)
      moveToCurrentTag()
    }

    const closeAllTags = (view) => {
      const { visitedViews } = tagsViewStore.delAllViews()
      if (affixTags.value.some(tag => tag.path === view.path)) {
        return
      }
      toLastView(visitedViews, view)
    }

    const toLastView = (visitedViewsList, view) => {
      const latestView = visitedViewsList.slice(-1)[0]
      if (latestView) {
        router.push(latestView.fullPath)
      } else {
        // now the default is to redirect to the home page if there is no tags-view,
        // you can adjust it according to your needs.
        if (view.name === 'Dashboard') {
          // to reload home page
          router.replace({ path: '/redirect' + view.fullPath })
        } else {
          router.push('/')
        }
      }
    }

    const openMenu = (tag, e) => {
      const menuMinWidth = 105
      const offsetLeft = document.getElementById('tags-view-container').getBoundingClientRect().left
      const offsetWidth = document.getElementById('tags-view-container').offsetWidth
      const maxLeft = offsetWidth - menuMinWidth
      const leftPos = e.clientX - offsetLeft + 15

      if (leftPos > maxLeft) {
        left.value = maxLeft
      } else {
        left.value = leftPos
      }

      top.value = e.clientY
      visible.value = true
      selectedTag.value = tag
    }

    const closeMenu = () => {
      visible.value = false
    }

    const handleScroll = () => {
      closeMenu()
    }

    watch(() => route.path, () => {
      addTags()
      moveToCurrentTag()
    })

    watch(visible, (value) => {
      if (value) {
        document.body.addEventListener('click', closeMenu)
      } else {
        document.body.removeEventListener('click', closeMenu)
      }
    })

    onMounted(() => {
      initTags()
      addTags()
    })

    return {
      visible,
      top,
      left,
      selectedTag,
      affixTags,
      scrollPane,
      tagRefs,
      visitedViews,
      isActive,
      isAffix,
      refreshSelectedTag,
      closeSelectedTag,
      closeOthersTags,
      closeAllTags,
      openMenu,
      handleScroll
    }
  }
}
</script>

<style lang="scss" scoped>
.tags-view-container {
  height: 34px;
  width: 100%;
  background: #fff;
  border-bottom: 1px solid #d8dce5;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, .12), 0 0 3px 0 rgba(0, 0, 0, .04);
  .tags-view-wrapper {
    .tags-view-item {
      display: inline-block;
      position: relative;
      cursor: pointer;
      height: 26px;
      line-height: 26px;
      border: 1px solid #d8dce5;
      color: #495060;
      background: #fff;
      padding: 0 8px;
      font-size: 12px;
      margin-left: 5px;
      margin-top: 4px;
      &:first-of-type {
        margin-left: 15px;
      }
      &:last-of-type {
        margin-right: 15px;
      }
      &.active {
        background-color: #42b983;
        color: #fff;
        border-color: #42b983;
        &::before {
          content: '';
          background: #fff;
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          position: relative;
          margin-right: 2px;
        }
      }
    }
  }
  .contextmenu {
    margin: 0;
    background: #fff;
    z-index: 3000;
    position: absolute;
    list-style-type: none;
    padding: 5px 0;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 400;
    color: #333;
    box-shadow: 2px 2px 3px 0 rgba(0, 0, 0, .3);
    li {
      margin: 0;
      padding: 7px 16px;
      cursor: pointer;
      &:hover {
        background: #eee;
      }
    }
  }
}
</style>

<style lang="scss">
//reset element css of el-icon-close
.tags-view-wrapper {
  .tags-view-item {
    .el-icon-close {
      width: 16px;
      height: 16px;
      vertical-align: 2px;
      border-radius: 50%;
      text-align: center;
      transition: all .3s cubic-bezier(.645, .045, .355, 1);
      transform-origin: 100% 50%;
      &:before {
        transform: scale(.6);
        display: inline-block;
        vertical-align: -3px;
      }
      &:hover {
        background-color: #b4bccc;
        color: #fff;
      }
    }
  }
}
</style>
