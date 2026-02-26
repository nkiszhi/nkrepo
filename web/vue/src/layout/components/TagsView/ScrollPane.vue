<template>
  <el-scrollbar ref="scrollContainer" class="scroll-container" @wheel.prevent="handleScroll">
    <slot />
  </el-scrollbar>
</template>

<script>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

const tagAndTagSpacing = 4 // tagAndTagSpacing

export default {
  name: 'ScrollPane',
  setup(props, { emit }) {
    const scrollContainer = ref(null)
    const leftPos = ref(0)

    const scrollWrapper = computed(() => {
      return scrollContainer.value?.$refs?.wrapRef
    })

    const handleScroll = (e) => {
      const eventDelta = e.wheelDelta || -e.deltaY * 40
      const $scrollWrapper = scrollWrapper.value
      if ($scrollWrapper) {
        $scrollWrapper.scrollLeft = $scrollWrapper.scrollLeft + eventDelta / 4
      }
    }

    const emitScroll = () => {
      emit('scroll')
    }

    const moveToTarget = (currentTag) => {
      const $container = scrollContainer.value?.$el
      const $containerWidth = $container?.offsetWidth || 0
      const $scrollWrapper = scrollWrapper.value
      if (!$scrollWrapper || !currentTag) return

      // Get all tag elements from parent
      const tagList = currentTag.$parent?.$refs?.tagRefs || []

      let firstTag = null
      let lastTag = null

      // find first tag and last tag
      if (tagList.length > 0) {
        firstTag = tagList[0]
        lastTag = tagList[tagList.length - 1]
      }

      if (firstTag === currentTag) {
        $scrollWrapper.scrollLeft = 0
      } else if (lastTag === currentTag) {
        $scrollWrapper.scrollLeft = $scrollWrapper.scrollWidth - $containerWidth
      } else {
        // find preTag and nextTag
        const currentIndex = tagList.findIndex(item => item === currentTag)
        if (currentIndex > 0 && currentIndex < tagList.length - 1) {
          const prevTag = tagList[currentIndex - 1]
          const nextTag = tagList[currentIndex + 1]

          if (nextTag?.$el && prevTag?.$el) {
            // the tag's offsetLeft after of nextTag
            const afterNextTagOffsetLeft = nextTag.$el.offsetLeft + nextTag.$el.offsetWidth + tagAndTagSpacing

            // the tag's offsetLeft before of prevTag
            const beforePrevTagOffsetLeft = prevTag.$el.offsetLeft - tagAndTagSpacing

            if (afterNextTagOffsetLeft > $scrollWrapper.scrollLeft + $containerWidth) {
              $scrollWrapper.scrollLeft = afterNextTagOffsetLeft - $containerWidth
            } else if (beforePrevTagOffsetLeft < $scrollWrapper.scrollLeft) {
              $scrollWrapper.scrollLeft = beforePrevTagOffsetLeft
            }
          }
        }
      }
    }

    onMounted(() => {
      const $scrollWrapper = scrollWrapper.value
      if ($scrollWrapper) {
        $scrollWrapper.addEventListener('scroll', emitScroll, true)
      }
    })

    onBeforeUnmount(() => {
      const $scrollWrapper = scrollWrapper.value
      if ($scrollWrapper) {
        $scrollWrapper.removeEventListener('scroll', emitScroll)
      }
    })

    return {
      scrollContainer,
      leftPos,
      handleScroll,
      moveToTarget
    }
  }
}
</script>

<style lang="scss" scoped>
.scroll-container {
  white-space: nowrap;
  position: relative;
  overflow: hidden;
  width: 100%;
  :deep(.el-scrollbar__bar) {
    bottom: 0px;
  }
  :deep(.el-scrollbar__wrap) {
    height: 49px;
  }
}
</style>
