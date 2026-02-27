<template>
  <span>{{ displayValue }}</span>
</template>

<script>
import { ref, watch, onMounted, onUnmounted } from 'vue'

export default {
  name: 'CountTo',
  props: {
    startVal: {
      type: Number,
      default: 0
    },
    endVal: {
      type: Number,
      required: true
    },
    duration: {
      type: Number,
      default: 2000
    },
    autoplay: {
      type: Boolean,
      default: true
    },
    decimals: {
      type: Number,
      default: 0
    },
    separator: {
      type: String,
      default: ','
    },
    prefix: {
      type: String,
      default: ''
    },
    suffix: {
      type: String,
      default: ''
    }
  },
  setup(props) {
    const displayValue = ref(props.startVal)
    let rafId = null
    let startTime = null
    let startVal = props.startVal

    const formatNumber = (num) => {
      const fixedNum = num.toFixed(props.decimals)
      const parts = fixedNum.split('.')
      parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, props.separator)
      return props.prefix + parts.join('.') + props.suffix
    }

    const easeOutQuad = (t, b, c, d) => {
      return -c * (t /= d) * (t - 2) + b
    }

    const count = (timestamp) => {
      if (!startTime) startTime = timestamp
      const progress = timestamp - startTime
      const currentVal = easeOutQuad(progress, startVal, props.endVal - startVal, props.duration)
      
      if (progress < props.duration) {
        displayValue.value = formatNumber(currentVal)
        rafId = requestAnimationFrame(count)
      } else {
        displayValue.value = formatNumber(props.endVal)
      }
    }

    const startCount = () => {
      startTime = null
      startVal = props.startVal
      if (rafId) {
        cancelAnimationFrame(rafId)
      }
      rafId = requestAnimationFrame(count)
    }

    watch(() => props.endVal, () => {
      if (props.autoplay) {
        startCount()
      }
    })

    onMounted(() => {
      if (props.autoplay) {
        startCount()
      }
    })

    onUnmounted(() => {
      if (rafId) {
        cancelAnimationFrame(rafId)
      }
    })

    return {
      displayValue
    }
  }
}
</script>
