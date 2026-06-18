export default {
  data() {
    return {
      $_sidebarElm: null,
      $_resizeHandler: null,
      $_resizeObserver: null,
      $_parentResizeObserver: null,
      $_resizeFrame: null
    }
  },
  mounted() {
    this.$_resizeHandler = () => {
      if (this.$_resizeFrame) return
      this.$_resizeFrame = window.requestAnimationFrame(() => {
        this.$_resizeFrame = null
        this.$_resizeChart()
      })
    }
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
    $_resizeChart() {
      if (!this.chart || !this.$el) {
        return
      }
      const rect = this.$el.getBoundingClientRect()
      if (rect.width <= 0 || rect.height <= 0) {
        return
      }
      this.chart.resize({
        width: Math.floor(rect.width),
        height: Math.floor(rect.height)
      })
      if (typeof this.setOptions === 'function') {
        this.setOptions(this.chartData)
      }
    },
    $_initElementResizeEvent() {
      if (!window.ResizeObserver || this.$_resizeObserver || !this.$el) {
        return
      }
      this.$_resizeObserver = new ResizeObserver(() => {
        this.$_resizeHandler()
      })
      this.$_resizeObserver.observe(this.$el)
      if (this.$el.parentElement) {
        this.$_parentResizeObserver = new ResizeObserver(() => {
          this.$_resizeHandler()
        })
        this.$_parentResizeObserver.observe(this.$el.parentElement)
      }
    },
    $_destroyElementResizeEvent() {
      if (this.$_resizeObserver) {
        this.$_resizeObserver.disconnect()
        this.$_resizeObserver = null
      }
      if (this.$_parentResizeObserver) {
        this.$_parentResizeObserver.disconnect()
        this.$_parentResizeObserver = null
      }
      if (this.$_resizeFrame) {
        window.cancelAnimationFrame(this.$_resizeFrame)
        this.$_resizeFrame = null
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
