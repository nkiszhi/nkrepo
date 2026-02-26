/**
 * How to use
 * <el-table height="100px" v-el-height-adaptive-table="{bottomOffset: 30}">...</el-table>
 * el-table height is must be set
 * bottomOffset: 30(default)   // The height of the table from the bottom of the page.
 */

const doResize = (el, binding) => {
  const { value } = binding
  const $table = el.__vueParentComponent?.proxy

  if (!$table) return

  const bottomOffset = (value && value.bottomOffset) || 30

  const height = window.innerHeight - el.getBoundingClientRect().top - bottomOffset

  // For Element Plus, we need to set the height differently
  if ($table.setHeight) {
    $table.setHeight(height)
  } else if ($table.layout) {
    $table.layout.setHeight(height)
    $table.doLayout()
  }
}

export default {
  mounted(el, binding) {
    el.resizeListener = () => {
      doResize(el, binding)
    }
    // Add resize listener
    window.addEventListener('resize', el.resizeListener)
    // Initial resize
    doResize(el, binding)
  },
  updated(el, binding) {
    doResize(el, binding)
  },
  unmounted(el) {
    window.removeEventListener('resize', el.resizeListener)
  }
}
