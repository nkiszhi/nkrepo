<template>
  <component :is="type" v-bind="linkProps(to)">
    <slot />
  </component>
</template>

<script>
import { computed } from 'vue'
import { isExternal } from '@/utils/validate'

export default {
  name: 'AppLink',
  props: {
    to: {
      type: String,
      required: true
    }
  },
  setup(props) {
    const isExternalLink = computed(() => isExternal(props.to))

    const type = computed(() => {
      if (isExternalLink.value) {
        return 'a'
      }
      return 'router-link'
    })

    const linkProps = (to) => {
      if (isExternalLink.value) {
        return {
          href: to,
          target: '_blank',
          rel: 'noopener'
        }
      }
      return {
        to: to
      }
    }

    return {
      type,
      linkProps
    }
  }
}
</script>
