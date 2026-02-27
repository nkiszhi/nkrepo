<template>
  <span>
    <svg-icon v-if="icon && !icon.includes('el-icon')" :icon-class="icon" />
    <i v-else-if="icon && icon.includes('el-icon')" :class="[icon, 'sub-el-icon']" />
    <span v-if="title" class="menu-title">{{ title }}</span>
  </span>
</template>

<script>
export default {
  name: 'MenuItem',
  props: {
    icon: {
      type: String,
      default: ''
    },
    title: {
      type: String,
      default: ''
    }
  }
}
</script>

<style scoped>
.sub-el-icon {
  color: currentColor;
  width: 1em;
  height: 1em;
  margin-right: 10px;
}

.menu-title {
  margin-left: 10px;
}
</style>
