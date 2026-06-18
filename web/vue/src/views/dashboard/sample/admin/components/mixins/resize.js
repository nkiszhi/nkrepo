import { debounce } from '@/utils'

export default {
  data() {
    return {
      $_sidebarElm: null,
      $_resizeHandler: null,
      $_resizeObserver: null
    }
  },
  mounted() {
    this.$_resizeHandler = debounce(() => {
      if (this.chart) {
        this.chart.resize()
        if (typeof this.setOptions === 'function') {
          this.setOptions(this.chartData)
        }
      }
    }, 100)
    this.$_initResizeEvent()
    this.$_initElementResizeEvent()
    this.$_initSidebarResizeEvent()
  },
  beforeUnmount() {
    this.$_destroyResizeEvent()
    this.$_destroyElementResizeEvent()
    this.$_destroySidebarResizeEvent()
  },
  // to fixed bug when cached by keep-alive
  // https://github.com/PanJiaChen/vue-element-admin/issues/2116
  activated() {
    this.$_initResizeEvent()
    this.$_initElementResizeEvent()
    this.$_initSidebarResizeEvent()
  },
  deactivated() {
    this.$_destroyResizeEvent()
    this.$_destroyElementResizeEvent()
    this.$_destroySidebarResizeEvent()
  },
  methods: {
    // use $_ for mixins properties
    // https://vuejs.org/v2/style-guide/index.html#Private-property-names-essential
    $_initResizeEvent() {
      window.addEventListener('resize', this.$_resizeHandler)
    },
    $_destroyResizeEvent() {
      window.removeEventListener('resize', this.$_resizeHandler)
    },
    $_initElementResizeEvent() {
      if (!window.ResizeObserver || this.$_resizeObserver || !this.$el) {
        return
      }
      this.$_resizeObserver = new ResizeObserver(() => {
        this.$_resizeHandler()
      })
      this.$_resizeObserver.observe(this.$el)
    },
    $_destroyElementResizeEvent() {
      if (this.$_resizeObserver) {
        this.$_resizeObserver.disconnect()
        this.$_resizeObserver = null
      }
    },
    $_sidebarResizeHandler(e) {
      if (e.propertyName === 'width') {
        this.$_resizeHandler()
      }
    },
    $_initSidebarResizeEvent() {
      this.$_sidebarElm = document.getElementsByClassName('sidebar-container')[0]
      this.$_sidebarElm && this.$_sidebarElm.addEventListener('transitionend', this.$_sidebarResizeHandler)
    },
    $_destroySidebarResizeEvent() {
      this.$_sidebarElm && this.$_sidebarElm.removeEventListener('transitionend', this.$_sidebarResizeHandler)
    }
  }
}
